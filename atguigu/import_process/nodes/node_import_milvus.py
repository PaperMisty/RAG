from pymilvus import CollectionSchema
from pymilvus import DataType
from atguigu.config.config import MilvusConfig
from atguigu.tool.milvus_client_tool import get_milvus_client
from atguigu.config.config import OUTPUT_DIR
from atguigu.tool.logger import logger
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
import json


import threading

milvus_lock = threading.Lock()


class NodeImportMilvus(NodeBase):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        chunks = state.get("chunk_dict_list")
        if not chunks:
            logger.error("chunks导入milvus节点, chunks不存在")
            raise ValueError("chunks导入milvus节点, chunks不存在")

        chunk_dim = len(chunks[0]["dense"])

        client = get_milvus_client()
        chunk_collection = MilvusConfig.chunks_collection

        schema = client.create_schema()
        with milvus_lock:
            if not client.has_collection(collection_name=chunk_collection):
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
                    field_name="section_title",
                    datatype=DataType.VARCHAR,
                    max_length=1000,
                ).add_field(
                    field_name="chunk_content", datatype=DataType.VARCHAR, max_length=65535
                ).add_field(
                    field_name="part", datatype=DataType.INT64, max_length=1000
                ).add_field(
                    # 页码（1 起 PDF 页序，0=未知）：CFQA 金标页级评测依赖此字段；
                    # 仅新建集合时生效，已存在的旧集合 schema 不受影响
                    field_name="page", datatype=DataType.INT64
                ).add_field(
                    field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=chunk_dim
                ).add_field(
                    field_name="sparse",
                    datatype=DataType.SPARSE_FLOAT_VECTOR,
                )
                # 创建索引
                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="dense",
                    index_name="dense_index",
                    index_type="AUTOINDEX",  # 根据数据量自定义索引算法 FLAT>>IVF_FLAT>>IVF_PQ>>HNSW
                    metric_type="COSINE",
                )
                index_params.add_index(
                    field_name="sparse",
                    index_name="sparse_index",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                    params={
                        "inverted_index_algo": "DAAT_MAXSCORE",
                        "normalize": True,
                        "quantization": "none",
                    },
                )
                # 创建表，增加并发防御
                try:
                    client.create_collection(chunk_collection, schema=schema, index_params=index_params)
                except Exception as e:
                    if "already exists" in str(e).lower() or "collectionexist" in str(e).lower():
                        logger.warning(f"Collection {chunk_collection} 已经在另一个任务中被创建，忽略此错误: {e}")
                    else:
                        raise e

        #  幂等性删除
        file_title = chunks[0]["file_title"]
        client.load_collection(chunk_collection)
        file_title = file_title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        client.delete(chunk_collection, filter=f"file_title == '{file_title}'")

        # 插入数据：先按目标集合的实际字段过滤 chunk 字典——切分侧新增的字段
        # （如 page）在旧集合（无该字段）上自动剔除，避免整批插入因未知字段
        # 失败；带 page 字段的新集合则完整写入。id 为自增主键，不入数据。
        schema_fields = {
            f["name"] for f in client.describe_collection(chunk_collection)["fields"]
        }
        insert_rows = [
            {k: v for k, v in chunk.items() if k in schema_fields and k != "id"}
            for chunk in chunks
        ]
        try:
            result = client.insert(chunk_collection, data=insert_rows)
        except Exception as e:
            # 入库兜底（用户 2026-10-23 规格）：chunk 超长触发 varchar
            # 上限硬报错时，不整份文档失败——把超长表格 chunk 按完整
            # 行拆分、重新向量化后重试一次。无可拆对象则原样抛错。
            remedied = self._split_oversized_chunks(chunks)
            if remedied is None:
                raise
            logger.warning(f"入库首次失败（{e}），超长表格 chunk 已按行拆分并重新向量化，重试入库")
            chunks = remedied
            # 保持幂等：重试前按 file_title 再删一次，避免首次部分写入造成重复
            client.delete(chunk_collection, filter=f"file_title == '{file_title}'")
            insert_rows = [
                {k: v for k, v in chunk.items() if k in schema_fields and k != "id"}
                for chunk in chunks
            ]
            result = client.insert(chunk_collection, data=insert_rows)
        logger.info(f"{chunk_collection}插入数据成功")

        # 给chunk的json文件回填id信息
        ids = result.get("ids")
        if ids:
            for i, chunk in enumerate(chunks):
                chunk["id"] = ids[i]

        return {"chunk_dict_list": chunks}

    def _split_oversized_chunks(self, chunks: list[dict]) -> list[dict] | None:
        """入库兜底：找出正文字节数超过 varchar 上限的 chunk，把其中
        的超长表格按完整行拆成 <=300 字片段并重新向量化；无可拆分
        对象时返回 None（调用方原样抛错）。

        拆分片段必须重新向量化：其 dense/sparse 与原 chunk 不对应，
        沿用旧向量会让检索内容与向量错位。向量化走 bgem3_tool 统一
        入口，文本组织（item_name + 换行 + chunk_content）与
        NodeBGEEmbedding 保持一致。
        """
        from atguigu.import_process.table_splitter import (
            expand_chunk_dict,
            split_table_content,
        )
        from atguigu.tool.bgem3_tool import get_bgem3_embedding

        limit_bytes = 65535  # 与本节点 schema 中 chunk_content 的 max_length 一致
        new_chunks: list[dict] = []
        for chunk in chunks:
            content = chunk.get("chunk_content", "")
            if len(content.encode("utf-8")) <= limit_bytes:
                new_chunks.append(chunk)
                continue
            fragments = split_table_content(content, 300)
            expanded = expand_chunk_dict(chunk, fragments)
            if len(expanded) == 1 and expanded[0] is chunk:
                return None  # 超限但无表可拆，补救不了
            new_chunks.extend(expanded)
        # 未变的 chunk 仍带 dense 字段；被拆出的片段已剔除旧向量，只补这些
        to_embed = [c for c in new_chunks if "dense" not in c]
        if not to_embed:
            return None
        batch_size = 64
        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [f"{c['item_name']}\n{c['chunk_content']}" for c in batch]
            vector_data = get_bgem3_embedding(texts)
            for idx, c in enumerate(batch):
                c["dense"] = vector_data["dense"][idx]
                c["sparse"] = vector_data["sparse"][idx]
        return new_chunks


if __name__ == "__main__":
    node = NodeImportMilvus()
    md_path = str(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new.md")
    with open(
        OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new_embedding_chunks.json", "r", encoding="utf-8"
    ) as f:
        chunks = json.load(f)
    init_state = {"chunk_dict_list": chunks}
    res = node(init_state)
    logger.info(res)
