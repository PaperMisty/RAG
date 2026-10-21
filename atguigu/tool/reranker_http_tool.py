"""【Colab 远程部署改造】Rerank 远程重排工具。

与原 atguigu.tool.reranker_tool.text_rerank（百炼 DashScope 云 API 版）保持
完全相同的函数签名与返回结构，业务节点（query_process/nodes/node_rerank.py）无感切换：

    text_rerank(query, documents, top_n=10)
        -> [{"index": i, "score": s}, ...]   # 按分数降序，最多 top_n 条
        # index 是【过滤空文档之后】的下标，与原实现语义一致（原实现先过滤再调 API）

后端由环境变量 EMBED_BACKEND 切换（与 embedding 共用一个开关，同一套远程服务）：
- tei（默认）：POST {BGE_RERANKER_URL}/rerank，
  请求 {"query": q, "texts": [docs...], "return_text": false}，
  返回 [{"index": i, "score": s}, ...] 且已按分数降序，按 index 字段还原即可；
- fastapi（回退）：POST {BGE_RERANKER_URL}/v1/rerank，
  请求 {"sentence_pairs": [[q, doc], ...]}，返回 {"scores": [...]}（与输入同序），
  这里组装成目标结构并按分数降序排列。
"""

import os

import requests

from atguigu.config.config import ReRankerConfig
from atguigu.tool.logger import logger

# 远程调用超时（秒），理由同 embedding 工具：防止远程服务假死挂住查询流程。
_TIMEOUT = float(os.getenv("RERANK_HTTP_TIMEOUT", "120"))

_session = requests.Session()


def _rerank_via_tei(query: str, valid_documents: list[str], actual_top_n: int) -> list:
    """走 TEI /rerank：返回已按分数降序，直接截取 top_n 并统一字段名。"""
    url = f"{ReRankerConfig.reranker_url.rstrip('/')}/rerank"
    resp = _session.post(
        url,
        json={"query": query, "texts": valid_documents, "return_text": False},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json()
    # TEI 的每项形如 {"index": 原始下标, "score": 分数}，下标即 valid_documents 的下标
    return [
        {"index": int(item["index"]), "score": float(item["score"])}
        for item in results[:actual_top_n]
    ]


def _rerank_via_fastapi(query: str, valid_documents: list[str], actual_top_n: int) -> list:
    """走老师 FastAPI 版 /v1/rerank：scores 与输入同序，组装后按分数降序截取 top_n。"""
    url = f"{ReRankerConfig.reranker_url.rstrip('/')}/v1/rerank"
    resp = _session.post(
        url,
        json={"sentence_pairs": [[query, doc] for doc in valid_documents]},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    scores = resp.json()["scores"]
    ranked = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:actual_top_n]


def text_rerank(query: str, documents: list[str], top_n: int = 10) -> list:
    """
    文本重排（远程自托管版，签名与原 DashScope 版一致）
    :param query: 查询文本
    :param documents: 文档列表
    :param top_n: 返回的文档数量
    :return: [{"index": 过滤后下标, "score": 分数}, ...]，按分数降序
    """
    # 与原实现一致：先过滤 None / 空字符串 / 非字符串，防止远程服务报错，
    # 且返回的 index 以过滤后的列表为准（node_rerank 的 documents 本就来自清洗后的结果，语义对齐）
    valid_documents = [doc for doc in documents if doc and isinstance(doc, str)]
    if not valid_documents:
        logger.warning("没有有效的文档可以进行重排序")
        return []

    # 确保 top_n 不超过有效文档数量
    actual_top_n = min(top_n, len(valid_documents))

    backend = (ReRankerConfig.embed_backend or "tei").lower()
    try:
        if backend == "fastapi":
            return _rerank_via_fastapi(query, valid_documents, actual_top_n)
        return _rerank_via_tei(query, valid_documents, actual_top_n)
    except Exception as e:
        # 与原实现一致：请求失败直接抛异常，不吞掉，让上层感知重排服务不可用
        logger.error(f"远程重排序请求失败: backend={backend}, error={e}")
        raise Exception(f"远程重排序请求异常: {e}") from e


if __name__ == "__main__":
    print(
        text_rerank(
            query="关于hak180烫金机如何使用？",
            documents=[
                "HAK 180 烫金机\n\n产品安全手册（简体中文）\n\n感谢您购买 HAK 180 烫金机。",
                "HAK 180 烫金机 是一款工业设备, 经常用于xxx ",
            ],
        )
    )
