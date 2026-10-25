# atguigu/query_process/state.py

from queue import Queue
from typing import TypedDict, List


class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。
    """

    task_id: str  # 任务ID
    session_id: str  # 会话ID
    message_id: str  # 消息ID
    q: Queue  # 任务队列

    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    dense_chunks: list  # Dense 稠密向量检索回来的切片
    sparse_chunks: list  # Sparse 稀疏向量检索回来的切片
    hyde_embedding_chunks: list  # 已向量化的假设性问题切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    key_words: List[str]  # 提取出的查询核心关键词
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_low_confidence: bool  # 是否低置信度（无主体识别）
