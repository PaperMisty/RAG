"""BGE-M3 向量统一入口（CFQA/Colab 接入）。

按 .env 的 EMBED_SOURCE 分发到三种实现，函数签名与返回结构完全一致：
    get_bgem3_embedding(texts) -> {"dense": [[...], ...],
                                   "sparse": [{token_id: weight, ...}, ...]}

- remote（默认）：bgem3_http_tool 远程 HTTP 版（TEI /embed_all 或 FastAPI 回退），
  即 Colab 远程部署改造后的现行行为，默认值保证不改变现有流程；
- local：bgem3_client_tool 本地版，进程内 BGEM3EmbeddingFunction 推理；
- colab：colab_client 网关版，走 Colab 聚合网关的 /embed + /embed_sparse。

导入在函数内懒加载：本地版会拉起 torch/pymilvus.model 的重依赖，
远程/colab 版只需要 requests，未选中的实现不应付出导入成本。
"""

from atguigu.config.config import EmbeddingConfig
from atguigu.tool.logger import logger


def get_bgem3_embedding(
    texts: list[str],
) -> dict[str, list[list[float]] | dict[int, list[float]]]:
    """把文本列表转为稀疏和稠密向量（按 EMBED_SOURCE 分发）。"""
    source = (EmbeddingConfig.embed_source or "remote").lower()
    if source == "local":
        from atguigu.tool.bgem3_client_tool import get_bgem3_embedding as impl
    elif source == "colab":
        from atguigu.tool.colab_client import colab_embed as impl
    else:
        if source != "remote":
            logger.warning(f"未知 EMBED_SOURCE={source}，回退 remote")
        from atguigu.tool.bgem3_http_tool import get_bgem3_embedding as impl
    return impl(texts)


if __name__ == "__main__":
    res = get_bgem3_embedding(texts=["hello world", "hello milvus"])
    print("dense dims:", [len(v) for v in res["dense"]])
    print("sparse sizes:", [len(s) for s in res["sparse"]])
