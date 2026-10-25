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
from typing import Any
from rich import print


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: QueryGraphState) -> dict:
        """职责: 执行 Re-Query 向量化与混合检索并上报观测层指标"""
        logger.info(f"【{self.name}】节点逻辑")
        task_id = state.get("task_id") or ""
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not item_names or not rewritten_query:
            logger.error(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")
            raise ValueError(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")

        # 执行 Re-Query 向量化检索（同时记录 Dense、Sparse 及混合检索详情至观测层）
        embedding_chunks, dense_chunks, sparse_chunks = self.search_chunks_with_details(
            rewritten_query, item_names, source="local", limit=10, task_id=task_id
        )
        return {
            "embedding_chunks": embedding_chunks,
            "dense_chunks": dense_chunks,
            "sparse_chunks": sparse_chunks,
        }

    @staticmethod
    def _fuse_dense_and_sparse(
        dense_chunks: list[dict], sparse_chunks: list[dict], limit: int = 10, dense_weight: float = 0.55, sparse_weight: float = 0.45
    ) -> list[dict]:
        """职责: 对Dense(Cosine)与Sparse(IP)检索结果做Min-Max分数归一化加权融合，解决Milvus跨度量尺度未归一化导致Sparse失效的问题"""
        def _norm_scores(chunks: list[dict]) -> dict[Any, float]:
            if not chunks:
                return {}
            scores = [float(c.get("score", 0.0) or 0.0) for c in chunks]
            s_max, s_min = max(scores), min(scores)
            span = s_max - s_min
            res_map = {}
            for idx, c in enumerate(chunks):
                cid = c.get("id")
                raw_s = float(c.get("score", 0.0) or 0.0)
                # 若分差极小则按排名衰减赋予 [0.5, 1.0] 归一化分，否则按 Min-Max 映射到 [0.3, 1.0]
                norm_s = (0.3 + 0.7 * (raw_s - s_min) / span) if span > 1e-6 else (1.0 - idx * 0.05)
                res_map[cid] = max(0.1, norm_s)
            return res_map

        dense_norm = _norm_scores(dense_chunks)
        sparse_norm = _norm_scores(sparse_chunks)

        merged: dict[Any, dict] = {}
        for rank_d, ch in enumerate(dense_chunks, 1):
            cid = ch.get("id")
            item = ch.copy()
            item["dense_rank"] = rank_d
            item["score"] = round(dense_weight * dense_norm.get(cid, 0.0), 4)
            merged[cid] = item

        for rank_s, ch in enumerate(sparse_chunks, 1):
            cid = ch.get("id")
            s_part = sparse_weight * sparse_norm.get(cid, 0.0)
            if cid in merged:
                merged[cid]["sparse_rank"] = rank_s
                merged[cid]["score"] = round(merged[cid]["score"] + s_part, 4)
            else:
                item = ch.copy()
                item["sparse_rank"] = rank_s
                item["score"] = round(s_part, 4)
                merged[cid] = item

        fused = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return fused[:limit]

    def search_chunks_with_details(
        self, query: str, item_names: list[str], source: str, limit: int = 10, task_id: str = ""
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """职责: 执行向量化并分别获取Dense排名、Sparse排名与混合检索排名及网络IO耗时"""
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

        # 3. 执行跨度量 Min-Max 归一化混合融合，确保稠密语义与稀疏关键词同时生效
        t_hyb = time.perf_counter()
        embedding_chunks = self._fuse_dense_and_sparse(dense_chunks, sparse_chunks, limit=limit)
        hyb_io_ms = round((time.perf_counter() - t_hyb) * 1000, 2)

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
            trace_collector.add_io_item(task_id, "retrieval", "Hybrid 归一化融合计算", hyb_io_ms, io_type="cpu")
            trace_collector.record_dense_sparse_retrieval(
                task_id=task_id,
                dense_chunks=dense_chunks,
                sparse_chunks=sparse_chunks,
                hybrid_chunks=embedding_chunks,
            )

        logger.info(f"Re-Query向量检索结果: {summarize_docs(embedding_chunks, limit=3)}")
        return embedding_chunks, dense_chunks, sparse_chunks

    def search_chunks(self, query: str, item_names: list[str], source: str, limit: int = 10) -> list[dict]:
        """职责: 对查询文本执行BGE-M3向量化与稠密+稀疏归一化混合检索"""
        vecs = get_bgem3_embedding([query])
        expr = f"item_name in {json.dumps(item_names, ensure_ascii=False)}"
        collection_name = MilvusConfig.chunks_collection
        output_fields = ["id", "file_title", "section_title", "chunk_content", "item_name"]

        dense_chunks, sparse_chunks, _, _ = search_dense_and_sparse_separately(
            collection_name=collection_name,
            dense_vec=vecs["dense"][0],
            sparse_vec=vecs["sparse"][0],
            expr=expr,
            limit=limit,
            output_fields=output_fields,
            source=source,
        )
        embedding_chunks = self._fuse_dense_and_sparse(dense_chunks, sparse_chunks, limit=limit)
        logger.info(f"向量混合检索结果: {summarize_docs(embedding_chunks, limit=3)}")
        return embedding_chunks


if __name__ == "__main__":
    node = NodeSearchEmbedding()
    # 标量检索是很死板的, 如果item_name差一个字,就会导致查找不到,全部过滤掉
    init_state = {"item_names": ["DK2509-10前后制动盘"], "rewritten_query": "DK2509-10前盘成本多少"}
    res = node(init_state)
    print(json_format(res))
