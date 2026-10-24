# atguigu/query_process/nodes/node_search_embedding.py

from atguigu.tool.json_format_tool import json_format
from atguigu.tool.milvus_client_tool import weighted_hybrid_search, search_dense_and_sparse_separately
from atguigu.config.config import MilvusConfig
from atguigu.tool.milvus_client_tool import create_reqs
# 【Colab远程部署改造】原本地模型版 bgem3_client_tool 改为远程 HTTP 版（函数签名与返回结构不变）
from atguigu.tool.bgem3_tool import get_bgem3_embedding
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger, summarize_docs
from observer.collector import trace_collector
import json
import time
from rich import print


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")
        task_id = state.get("task_id") or ""
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not item_names or not rewritten_query:
            logger.error(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")
            raise ValueError(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")
        embedding_chunks, dense_chunks, sparse_chunks = self.search_chunks_with_details(
            rewritten_query, item_names, source="local", limit=10, task_id=task_id
        )
        return {
            "embedding_chunks": embedding_chunks,
            "dense_chunks": dense_chunks,
            "sparse_chunks": sparse_chunks,
        }

    def search_chunks_with_details(
        self, query: str, item_names: list[str], source: str, limit: int = 10, task_id: str = ""
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """职责: 执行向量化并分别获取Dense排名、Sparse排名与混合检索排名及网络IO耗时"""
        t_start = time.perf_counter()

        # 1. 将重写问题转换成 embedding 向量 (记录 BGE-M3 HTTP IO)
        t_emb = time.perf_counter()
        vecs = get_bgem3_embedding([query])
        emb_io_ms = round((time.perf_counter() - t_emb) * 1000, 2)

        expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}"
        collection_name = MilvusConfig.chunks_collection
        output_fields = ["id", "file_title", "section_title", "chunk_content", "item_name"]

        # 2. 分别执行 Dense 稠密检索和 Sparse 稀疏检索以供观测层按排名展示
        dense_chunks, sparse_chunks, dense_io_ms, sparse_io_ms = search_dense_and_sparse_separately(
            collection_name=collection_name,
            dense_vec=vecs["dense"][0],
            sparse_vec=vecs["sparse"][0],
            expr=expr,
            limit=limit,
            output_fields=output_fields,
            source=source,
        )

        # 3. 执行原有加权混合检索
        reqs = create_reqs(
            dense_data=vecs["dense"][0],
            sparse_data=vecs["sparse"][0],
            dense_ann_field="dense",
            sparse_ann_field="sparse",
            expr=expr,
        )
        t_hyb = time.perf_counter()
        res = weighted_hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=[0.8, 0.2],
            limit=limit,
            output_fields=output_fields,
        )
        hyb_io_ms = round((time.perf_counter() - t_hyb) * 1000, 2)

        embedding_chunks = []
        for search_result in res[0]:
            embedding_chunks.append({**search_result["entity"], "score": search_result["distance"], "source": source})

        # 4. 写入观测层收集器
        if task_id:
            trace_collector.add_io_item(task_id, "dense", "BGE-M3 向量化 HTTP IO", emb_io_ms, io_type="network")
            trace_collector.add_io_item(task_id, "dense", "Milvus Dense 检索 IO", dense_io_ms, io_type="db")
            trace_collector.set_module_total_ms(task_id, "dense", emb_io_ms + dense_io_ms)

            trace_collector.add_io_item(task_id, "sparse", "BGE-M3 稀疏向量复用", 0.0, io_type="cpu", detail="与Dense同次返回")
            trace_collector.add_io_item(task_id, "sparse", "Milvus Sparse 检索 IO", sparse_io_ms, io_type="db")
            trace_collector.set_module_total_ms(task_id, "sparse", sparse_io_ms)

            trace_collector.add_io_item(task_id, "retrieval", "BGE-M3 Query 向量化 IO", emb_io_ms, io_type="network")
            trace_collector.add_io_item(task_id, "retrieval", "Milvus Dense 检索 IO", dense_io_ms, io_type="db")
            trace_collector.add_io_item(task_id, "retrieval", "Milvus Sparse 检索 IO", sparse_io_ms, io_type="db")
            trace_collector.add_io_item(task_id, "retrieval", "Milvus Hybrid 检索 IO", hyb_io_ms, io_type="db")
            trace_collector.record_dense_sparse_retrieval(
                task_id=task_id,
                dense_chunks=dense_chunks,
                sparse_chunks=sparse_chunks,
                hybrid_chunks=embedding_chunks,
            )

        logger.info(f"Re-Query向量检索结果: {summarize_docs(embedding_chunks, limit=3)}")
        return embedding_chunks, dense_chunks, sparse_chunks

    def search_chunks(self, query, item_names, source, limit=10):
        # 将重写问题转换成embedding向量
        vecs = get_bgem3_embedding([query])
        # 标量搜索依然很重要, 不过expr逐渐推荐用filter代替了
        expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}"  # json.dumps 自身已具备完美的自动转义机制
        reqs = create_reqs(
            dense_data=vecs["dense"][0],
            sparse_data=vecs["sparse"][0],
            dense_ann_field="dense",
            sparse_ann_field="sparse",
            expr=expr,
        )
        collection_name = MilvusConfig.chunks_collection
        res = weighted_hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=[0.8, 0.2],
            limit=limit,
            output_fields=["id", "file_title", "section_title", "chunk_content", "item_name"],
        )
        # print(json_format(res))

        # 组装检索出的chunk信息
        embedding_chunks = []
        for search_result in res[0]:
            embedding_chunks.append({**search_result["entity"], "score": search_result["distance"], "source": source})

        # =========== 展示示例 ==========
        logger.info(f"Re-Query向量检索结果: {summarize_docs(embedding_chunks, limit=3)}")
        return embedding_chunks


if __name__ == "__main__":
    node = NodeSearchEmbedding()
    # 标量检索是很死板的, 如果item_name差一个字,就会导致查找不到,全部过滤掉
    init_state = {"item_names": ["DK2509-10前后制动盘"], "rewritten_query": "DK2509-10前盘成本多少"}
    res = node(init_state)
    print(json_format(res))
