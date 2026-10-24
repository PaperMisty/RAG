from atguigu.tool.mongo_client_tool import clear_history
from atguigu.tool.mongo_client_tool import add_or_update_data
from langgraph.constants import END
from langgraph.graph import StateGraph

from atguigu.query_process.nodes.node_answer_output import NodeAnswerOutput
from atguigu.query_process.nodes.node_item_name_confirm import NodeItemNameConfirm
from atguigu.query_process.nodes.node_rerank import NodeRerank
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from atguigu.query_process.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from atguigu.query_process.nodes.node_web_search_mcp import NodeWebSearchMcp
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger
from observer.collector import trace_collector
import time
import uuid


class QueryMainGraphRunner:
    def __init__(self):
        self.builder = StateGraph(state_schema=QueryGraphState)
        self.add_nodes()
        self.add_edges()
        self.graph = None

    def add_nodes(self):
        self.builder.add_node(NodeItemNameConfirm.name, NodeItemNameConfirm())
        self.builder.add_node(NodeSearchEmbedding.name, NodeSearchEmbedding())
        self.builder.add_node(NodeSearchEmbeddingHyde.name, NodeSearchEmbeddingHyde())
        self.builder.add_node(NodeWebSearchMcp.name, NodeWebSearchMcp())
        self.builder.add_node(NodeRrf.name, NodeRrf())
        self.builder.add_node(NodeRerank.name, NodeRerank())
        self.builder.add_node(NodeAnswerOutput.name, NodeAnswerOutput())

    def after_item_name_confirm_router(self, state: QueryGraphState):
        answer = state.get("answer", "")
        is_low_confidence = state.get("is_low_confidence", False)
        if answer:
            # 中置信度多个选项，跳转至澄清回答
            return NodeAnswerOutput.name
        elif is_low_confidence:
            # 低置信度，只路由去网络搜索节点
            logger.info("【路由】主体匹配置信度低，只进行网络搜索")
            return NodeWebSearchMcp.name
        else:
            # 高置信度，本地向量 + 网络搜索并行检索
            return [NodeSearchEmbeddingHyde.name, NodeSearchEmbedding.name, NodeWebSearchMcp.name]

    def after_web_search_router(self, state: QueryGraphState):
        is_low_confidence = state.get("is_low_confidence", False)
        if is_low_confidence:
            # 低置信度，网络搜索后直接跳转到输出生成节点
            return NodeAnswerOutput.name
        else:
            # 正常高置信度，网络搜索后进入 RRF 排序融合
            return NodeRrf.name

    def add_edges(self):
        self.builder.set_entry_point(NodeItemNameConfirm.name)
        self.builder.add_conditional_edges(NodeItemNameConfirm.name, self.after_item_name_confirm_router)
        self.builder.add_edge(NodeSearchEmbeddingHyde.name, NodeRrf.name)
        self.builder.add_edge(NodeSearchEmbedding.name, NodeRrf.name)
        self.builder.add_conditional_edges(NodeWebSearchMcp.name, self.after_web_search_router)
        self.builder.add_edge(NodeRrf.name, NodeRerank.name)
        self.builder.add_edge(NodeRerank.name, NodeAnswerOutput.name)
        self.builder.add_edge(NodeAnswerOutput.name, END)

    def run(self, state):
        if self.graph is None:
            self.graph = self.builder.compile()
        if not state.get("task_id"):
            state["task_id"] = str(uuid.uuid4())
        task_id = state["task_id"]
        if trace_collector.get_trace(task_id) is None:
            trace_collector.start_trace(
                task_id=task_id,
                original_query=state.get("original_query", ""),
                session_id=state.get("session_id", ""),
            )
        t0 = time.perf_counter()
        try:
            result = self.graph.invoke(state)
            trace_collector.finish_trace(
                task_id=task_id,
                status="completed",
                total_duration_ms=(time.perf_counter() - t0) * 1000,
            )
            return result
        except Exception as exc:
            trace_collector.finish_trace(
                task_id=task_id,
                status="failed",
                error_message=str(exc)[:300],
                total_duration_ms=(time.perf_counter() - t0) * 1000,
            )
            raise

    @classmethod
    def create_and_run(cls, state):
        return cls().run(state)


if __name__ == "__main__":
    import time

    session_id = "test_001"
    # 清理记录
    clear_history(session_id)
    # 添加数据
    data_dict = {
        "session_id": session_id,
        "role": "user",
        "text": "咨询下烫金机",  # 和万用表。
        "rewritten_query": None,
        "item_names": None,
        "ts": time.time(),
    }
    add_or_update_data(data_dict)

    data_dict = {
        "session_id": session_id,
        "role": "assistant",
        "text": "您好。请问是哪个型号",
        "rewritten_query": None,
        "item_names": None,
        "ts": time.time(),
    }
    add_or_update_data(data_dict)

    data_dict = {
        "session_id": session_id,
        "role": "user",
        "text": "hak180",  # 和RS-12
        "rewritten_query": None,
        "item_names": None,
        "ts": time.time(),
    }
    add_or_update_data(data_dict)

    data_dict = {
        "session_id": session_id,
        "role": "assistant",
        "text": "具体有什么问题呢？",
        "rewritten_query": None,
        "item_names": None,
        "ts": time.time(),
    }
    add_or_update_data(data_dict)
    # 初始化图状态
    init_state = {"session_id": "test_001", "original_query": "咋用？"}
    result = QueryMainGraphRunner.create_and_run(init_state)

    print(json_format(result))
