# atguigu/import_process/nodes/node_bge_embedding.py
from atguigu.tool.validate_path import validate_path
# 【Colab远程部署改造】原本地模型版 bgem3_client_tool 改为远程 HTTP 版（函数签名与返回结构不变）
from atguigu.tool.bgem3_tool import get_bgem3_embedding
from atguigu.tool.json_format_tool import json_format
from atguigu.config.config import OUTPUT_DIR
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
import json
from pathlib import Path
from atguigu.tool.logger import logger


class NodeBGEEmbedding(NodeBase):
    """
    混合向量化节点：使用 BGE-M3 模型将每个chunk_content文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """将chunk内容进行向量化,并保存到json文件中

        Args:
            state (ImportGraphState):

        Raises:
            ValueError:

        Returns:
            ImportGraphState:
        """
        chunks = state.get("chunk_dict_list")
        if not chunks:
            logger.error("chunk_dict_list无内容,必须传入值,进行向量化")
            raise ValueError("chunk_dict_list无内容,必须传入值,进行向量化")

        # 按批次拼接chunks中的内容，
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            chunk_temp = chunks[i : i + batch_size]
            content_list = [f"{chunk['item_name']}\n{chunk['chunk_content']}" for chunk in chunk_temp]
            # 通过bgem3模型获取向量
            vector_data = get_bgem3_embedding(content_list)
            for idx, chunk in enumerate(chunk_temp):
                chunk["dense"] = vector_data["dense"][idx]  # 利用[列表]和[字典值]可变性增加字段
                chunk["sparse"] = vector_data["sparse"][idx]

        # 新chunk写进json文件
        md_path_obj = Path(state.get("md_path", ""))
        md_path_obj = validate_path(md_path_obj, "error")

        with open(md_path_obj.parent / f"{md_path_obj.stem}_embedding_chunks.json", "w", encoding="utf-8") as f:
            f.write(json_format(chunks))
        logger.info(f"{str(md_path_obj)}已经写入chunks内容")

        return {"chunk_dict_list": chunks}


if __name__ == "__main__":
    node = NodeBGEEmbedding()
    md_path = str(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new.md")
    with open(OUTPUT_DIR / "hak180产品安全手册" / "hak180产品安全手册_new_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    init_state = {"chunk_dict_list": chunks, "md_path": md_path}
    res = node(init_state)
    logger.info(json_format(res))
