# atguigu/query_process/nodes/node_search_embedding_hyde.py

from atguigu.config.config import PromptConfig
from atguigu.config.config import LLMConfig, TextLLMConfig  # 【Colab远程部署改造】文本模型走网关配置
from atguigu.config.config import MilvusConfig
from langchain.chat_models import init_chat_model
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.tool.json_format_tool import json_format
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger, summarize_docs
from observer.collector import trace_collector
import time


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        """
        重写问题给到LLM输出Hybird答案,追加问题进行检索数据库
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        task_id = state.get("task_id") or ""
        t_start = time.perf_counter()
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        if not item_names or not rewritten_query:
            logger.error(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")
            raise ValueError(f"缺少主体名称或改写后的问题{item_names=},{rewritten_query=}")

        llm = init_chat_model(
            model=TextLLMConfig.default_model,
            model_provider=TextLLMConfig.model_provider,
            base_url=TextLLMConfig.base_url,
            api_key=TextLLMConfig.api_key,
            extra_body=TextLLMConfig.extra_body,
            temperature=LLMConfig.temperature,
        )
        msg = [
            {"role": "system", "content": "你是专业AI助手，严谨简洁，不要编造信息"},
            {"role": "user", "content": PromptConfig.HYDE_PROMPT.format(rewritten_query=rewritten_query)},
        ]

        t_llm = time.perf_counter()
        hybrid_answer = llm.invoke(input=msg).content
        llm_io_ms = round((time.perf_counter() - t_llm) * 1000, 2)

        t_search = time.perf_counter()
        hyde_embedding_chunks = NodeSearchEmbedding().search_chunks(
            f"问题:{rewritten_query}\n答案:{hybrid_answer}", item_names, source="local"
        )
        search_io_ms = round((time.perf_counter() - t_search) * 1000, 2)

        if task_id:
            trace_collector.add_io_item(task_id, "hyde", "HyDE LLM 生成 IO", llm_io_ms, io_type="llm")
            trace_collector.add_io_item(task_id, "hyde", "HyDE 向量化与Milvus检索 IO", search_io_ms, io_type="network")
            trace_collector.set_module_total_ms(task_id, "hyde", (time.perf_counter() - t_start) * 1000)
            trace_collector.add_io_item(task_id, "retrieval", "HyDE LLM 假想文档 IO", llm_io_ms, io_type="llm")
            trace_collector.record_hyde_retrieval(task_id, hybrid_answer, hyde_embedding_chunks)

        # =========== 展示示例 ==========
        logger.info(f"HyDE检索结果: {summarize_docs(hyde_embedding_chunks, limit=3)}")
        # return state
        return {"hyde_embedding_chunks": hyde_embedding_chunks}


if __name__ == "__main__":
    node = NodeSearchEmbeddingHyde()
    init_state = {"item_names": ["兄弟HAK180烫金机"], "rewritten_query": "兄弟HAK180烫金机咋用？"}
    res = node(init_state)
    print(json_format(res))
