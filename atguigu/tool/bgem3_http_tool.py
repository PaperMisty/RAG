"""【Colab 远程部署改造】BGE-M3 远程向量化工具。

与本地版 atguigu.tool.bgem3_client_tool.get_bgem3_embedding 保持完全相同的
函数签名与返回结构，业务节点无感切换：

    get_bgem3_embedding(texts) -> {"dense": [[...], ...],
                                   "sparse": [{token_id: weight, ...}, ...]}

后端由环境变量 EMBED_BACKEND 切换（在 .env 中配置）：
- tei（默认）：POST {BGE_M3_URL}/embed_all，一次返回 dense + sparse，
  这是项目 Milvus 混合检索（dense + sparse 双路）能成立的前提；
- fastapi（回退）：POST {BGE_M3_URL}/v1/embeddings，老师 FastAPI 版服务，
  请求 {"embedding_documents": [...]}，响应即目标结构，直接透传。
"""

import os

import requests

from atguigu.config.config import EmbeddingConfig
from atguigu.tool.logger import logger

# 远程调用超时（秒）：本地版是进程内推理没有超时概念，远程版必须设，
# 否则 Colab 侧服务假死会把整个查询流程挂住。
_TIMEOUT = float(os.getenv("EMBED_HTTP_TIMEOUT", "120"))

_session = requests.Session()


def _sparse_item_to_dict(item) -> dict:
    """把一条 sparse 向量的各种常见表示统一转成 {int token_id: float weight}。

    兼容三种表示：
    1. [{"index": i, "value": v}, ...] 列表（TEI 的典型输出）
    2. {"indices": [...], "values": [...]} 双列表字典
    3. {token_id: weight} 字典（JSON 里键会变成字符串，这里转回 int，
       与本地版 BGEM3EmbeddingFunction 输出的 int 键保持一致，Milvus 侧才认）
    """
    if isinstance(item, dict) and "indices" in item and ("values" in item or "weights" in item):
        # Colab 网关（FlagEmbedding 实现）的 /embed_sparse 用 weights 命名权重，
        # 与 TEI 惯例的 values 等价，两种都认（2026-10-23 联调实测确认）
        values = item["values"] if "values" in item else item["weights"]
        return {int(i): float(v) for i, v in zip(item["indices"], values)}
    if isinstance(item, dict):
        return {int(k): float(v) for k, v in item.items()}
    if isinstance(item, list):
        return {int(pair["index"]): float(pair["value"]) for pair in item}
    raise ValueError(f"无法识别的 sparse 表示类型: {type(item)}")


def _parse_tei_embed_all(data) -> dict:
    """解析 TEI /embed_all 的响应，转换为项目统一结构。

    TODO（今晚联调确认）：TEI /embed_all 的确切 JSON 形状未经实测，
    联调时按实际返回只改本函数这一处，其余代码不用动。
    当前按最可能的形状实现：顶层为 list，每个元素对应一条输入文本，
    元素内含 "dense"（向量列表）与 "sparse"（见 _sparse_item_to_dict 的兼容表示），
    可能还带 "colbert_vecs"（本项目不用，忽略）。
    若顶层是 dict，则尝试取 data["embeddings"] / data["data"] 作为列表。
    """
    items = data
    if isinstance(data, dict):
        items = data.get("embeddings") or data.get("data")
        if items is None:
            raise ValueError(
                f"TEI /embed_all 返回 dict 但无 embeddings/data 字段，实际字段: {list(data.keys())}"
            )
    if not isinstance(items, list):
        raise ValueError(f"TEI /embed_all 返回结构无法识别: {type(data)}")

    dense_list = []
    sparse_list = []
    for item in items:
        if not isinstance(item, dict) or "dense" not in item:
            raise ValueError(f"TEI /embed_all 单条结果缺少 dense 字段: {type(item)}")
        dense_list.append([float(x) for x in item["dense"]])
        sparse_list.append(_sparse_item_to_dict(item.get("sparse", {})))
    return {"dense": dense_list, "sparse": sparse_list}


def _embed_via_tei(texts: list[str]) -> dict:
    """走 TEI /embed_all：一次请求同时拿 dense + sparse。"""
    url = f"{EmbeddingConfig.bge_m3_url.rstrip('/')}/embed_all"
    # truncate=True 对齐本地版 encode_documents 的静默截断行为，
    # 避免超长 chunk 直接触发 TEI 报错；normalize=True 对齐本地版默认归一化。
    resp = _session.post(
        url,
        json={"inputs": texts, "truncate": True, "normalize": True},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return _parse_tei_embed_all(resp.json())


def _embed_via_fastapi(texts: list[str]) -> dict:
    """走老师 FastAPI 版服务 /v1/embeddings：响应即目标结构，透传即可。"""
    url = f"{EmbeddingConfig.bge_m3_url.rstrip('/')}/v1/embeddings"
    resp = _session.post(url, json={"embedding_documents": texts}, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # FastAPI 版 sparse 的键经 JSON 往返后是字符串，转回 int 与本地版保持一致
    return {
        "dense": data["dense"],
        "sparse": [{int(k): float(v) for k, v in sparse_item.items()} for sparse_item in data["sparse"]],
    }


def get_bgem3_embedding(
    texts: list[str],
) -> dict[str, list[list[float]] | dict[int, list[float]]]:
    """把文本列表转为稀疏和稠密向量（远程版，签名与本地版完全一致）。

    Args:
        texts (list[str]): 文本列表

    Returns:
        dict: {"dense": 稠密向量列表, "sparse": 稀疏向量字典列表}
    """
    backend = (EmbeddingConfig.embed_backend or "tei").lower()
    if backend == "fastapi":
        return _embed_via_fastapi(texts)
    return _embed_via_tei(texts)


if __name__ == "__main__":
    res = get_bgem3_embedding(texts=["hello world", "hello milvus"])
    print("dense dims:", [len(v) for v in res["dense"]])
    print("sparse sizes:", [len(s) for s in res["sparse"]])
