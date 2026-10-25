# atguigu/import_process/nodes/node_item_name_recognition.py
from atguigu.config.config import PromptConfig
from pymilvus import MilvusClient
from atguigu.config.config import MilvusConfig
# 向量化走统一分发 bgem3_tool（按 EMBED_SOURCE 选 local/remote/colab，与 node_bge_embedding 一致）；
# 原写死 bgem3_http_tool 远程版，EMBED_SOURCE=local 的本地入库路径下主体向量无处可取，故改走分发。
from atguigu.tool.bgem3_tool import get_bgem3_embedding
from pymilvus import CollectionSchema
from pymilvus import DataType
from atguigu.tool.milvus_client_tool import get_milvus_client
from langchain.chat_models import init_chat_model
import os
from atguigu.config.config import LLMConfig, TextLLMConfig  # 【Colab远程部署改造】文本模型走网关配置
from atguigu.tool.validate_path import validate_path
from atguigu.tool.logger import logger
from atguigu.tool.json_format_tool import json_format
from atguigu.config.config import OUTPUT_DIR
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
import json
import threading
from pathlib import Path

milvus_lock = threading.Lock()


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        chunks = state.get("chunk_dict_list")
        if not chunks:
            logger.error(f"chunk无内容,必须传入值,进行主体识别")
            raise ValueError(f"chunk无内容,必须传入值,进行主体识别")
        md_path_obj = Path(state.get("md_path", ""))
        md_path_obj = validate_path(md_path_obj, "error")

        file_title = chunks[0]["file_title"]
        if not file_title:
            logger.error("文件标题为空")
            raise ValueError("文件标题为空")

        # 1. 通过llm总结获取主体名称和文件标题
        item_name = self.get_item_name(file_title, chunks)
        # 2. 获取milvus连接和表
        client, collection = self.get_milvus_collection(file_title)
        # 3. 向量化主体名称并插入milvus表
        self.get_item_vecs(item_name, file_title, client, collection)
        # 4. chunks追加item_name, 并写入json
        chunks = self.write_chunk_json(chunks, item_name, md_path_obj)
        return {"chunk_dict_list": chunks}

    def get_item_name(self, file_title, chunks: list[dict]) -> tuple[str, str]:
        """获取主体名称和文件标题

        Args:
            chunks (list[dict]): 文档内容切片

        Raises:
            ValueError: 主体识别失败，必须传入主体名称

        Returns:
            tuple[str, str]: 主体名称和文件标题
        """
        # 切片固定长度chunks,输给LLM总结
        MAX_LENGTH = 10000
        content_example = json.dumps(chunks[:10], ensure_ascii=False)[:MAX_LENGTH]

        llm = init_chat_model(
            model=TextLLMConfig.default_model,
            model_provider="openai",
            base_url=TextLLMConfig.base_url,
            api_key=TextLLMConfig.api_key,
            extra_body=TextLLMConfig.extra_body,
            temperature=LLMConfig.temperature,
        )
        msg = [
            {
                "role": "user",
                "content": PromptConfig.ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                    file_title=file_title, context=content_example
                ),
            }
        ]
        res = llm.invoke(input=msg)
        item_name = res.content.strip().replace(" ", "")
        logger.info(f"主体识别结果: {item_name}")

        if not item_name:
            logger.debug("主体识别失败，默认采用文件名作为主体名称")
            item_name = file_title  # 主体item_name,是作为file_title可能无意义的优化版本
        return item_name

    def get_milvus_collection(self, file_title: str) -> tuple[MilvusClient, str]:
        """获取milvus连接和表

        Args:
            file_title (str): 文件标题

        Returns:
            tuple[MilvusClient, str]: milvus连接和表
        """
        # 创建milvus表
        client = get_milvus_client()
        collection = MilvusConfig.item_name_collection
        with milvus_lock:
            if not client.has_collection(collection_name=collection):
                schema: CollectionSchema = client.create_schema(auto_id=True)
                schema.add_field(
                    field_name="id",
                    datatype=DataType.INT64,
                    is_primary=True,
                ).add_field(
                    field_name="item_name",
                    datatype=DataType.VARCHAR,
                    max_length=1000,
                ).add_field(
                    field_name="file_title",
                    datatype=DataType.VARCHAR,
                    max_length=1000,
                ).add_field(
                    field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024
                ).add_field(
                    field_name="sparse",
                    datatype=DataType.SPARSE_FLOAT_VECTOR,
                )
                # 创建索引
                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="dense",
                    index_name="dense_index",
                    index_type="IVF_FLAT",
                    metric_type="COSINE",
                    params={"nlist": 1000, "nprobe": 100},
                )
                index_params.add_index(
                    field_name="sparse",
                    index_name="sparse_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                    params={
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        # 高效的稀疏检索算法
                        "normalize": True,
                        # ↑ L2 归一化，让内积 (IP) 等价于余弦相似度
                        "quantization": "none",
                        # ↑ 关闭量化，保持原始精度
                    },
                )
                try:
                    client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
                except Exception as e:
                    if "already exists" in str(e).lower() or "collectionexist" in str(e).lower():
                        logger.warning(f"Collection {collection} 已经在另一个任务中被创建，忽略此错误: {e}")
                    else:
                        raise e

        # 幂等性删除表部分内容 ; TODO 可以考虑将md文档hash为依据进行幂等性删除
        client.load_collection(collection_name=collection)
        safe_file_title = file_title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')  # 防止注入
        client.delete(collection, filter=f"file_title=='{safe_file_title}'")
        return client, collection

    def get_item_vecs(self, item_name: str, file_title: str, client: MilvusClient, collection: str):
        """向量化主体名称并插入milvus表

        Args:
            item_name (str): 主体名称
            file_title (str): 文件标题
            client (MilvusClient): milvus连接
            collection (str): milvus表名
        """
        # item_name向量化
        vecs = get_bgem3_embedding([item_name])
        data = {
            "file_title": file_title,
            "item_name": item_name,
            "dense": vecs["dense"][0],
            "sparse": vecs["sparse"][0],
        }
        client.insert(collection, data)
        logger.info(f"{file_title}内容插入成功")

    def write_chunk_json(self, chunks, item_name: str, md_path_obj: Path) -> list[dict]:
        """追加item_name, 并写入json

        Args:
            chunks (list[dict]): 切分后的文档内容
            item_name (str): 主体名称
            md_path_obj (Path): md文件路径

        Returns:
            list[dict]: 追加item_name的chunks数据
        """
        for chunk in chunks:
            chunk["item_name"] = item_name
        chunks_json_path = str(md_path_obj.parent) + "/" + md_path_obj.stem + "_chunks.json"  # 直接覆写了原始数据
        with open(
            chunks_json_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json_format(chunks))
        logger.info(f"切分后的文档内容写入{chunks_json_path}")
        return chunks


if __name__ == "__main__":
    node = NodeItemNameRecognition()
    file_path = str(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new_chunks.json")
    md_path = str(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new.md")
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        chunk_data = json.load(f)
    init_state = {"chunk_data": chunk_data, "md_path": md_path}

    res = node(init_state)
    # logger.info(json_format(res))
