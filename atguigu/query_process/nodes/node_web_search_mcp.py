# atguigu/query_process/nodes/node_web_search_mcp.py
"""Web search node, backed by Tavily Search API.

This replaces the former Bailian MCP path: the agents MCP client handshake
against the DashScope endpoint timed out at exactly 10s/60s, and no
fallback to that MCP path is kept here. If Tavily fails, this node returns
an empty list; the answer node decides what to do (low confidence -> fixed
sentence, high confidence -> pure KB answer).
"""

import time

import httpx

from atguigu.config.config import TavilyConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger
from observer.collector import trace_collector


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用 Tavily 搜索补充信息。
    （类名/文件名保留 node_web_search_mcp，避免改动图路由；后端已切 Tavily。）
    """

    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState):
        logger.info(f"【{self.name}】节点逻辑")
        task_id = state.get("task_id") or ""
        t_start = time.perf_counter()
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            logger.error(f"【{self.name}】节点的处理数据为空")
            raise ValueError(f"【{self.name}】节点的处理数据为空")
        try:
            with trace_collector.track_io(task_id, "web", "Tavily Search HTTP IO", io_type="network"):
                docs = self.tavily_search(rewritten_query)
            web_io_ms = round((time.perf_counter() - t_start) * 1000, 2)
            trace_collector.set_module_total_ms(task_id, "web", web_io_ms)
            trace_collector.add_io_item(task_id, "retrieval", "Tavily 网络搜索 IO", web_io_ms, io_type="network")
            trace_collector.record_web_retrieval(task_id, docs)
        except Exception as e:
            web_io_ms = round((time.perf_counter() - t_start) * 1000, 2)
            trace_collector.set_module_total_ms(task_id, "web", web_io_ms)
            trace_collector.add_io_item(
                task_id, "retrieval", "Tavily 网络搜索 IO (失败)", web_io_ms, io_type="network", status="error"
            )
            trace_collector.record_web_retrieval(task_id, [])
            # 不抛错、不回头走百炼 MCP：失败一律返回空列表，由答案节点按置信度分流。
            logger.warning(f"【{self.name}】Tavily 搜索失败，返回空列表: {type(e).__name__}")
            return {"web_search_docs": []}
        if not docs:
            logger.warning(f"【{self.name}】Tavily 搜索结果为空，返回空列表")
        return {"web_search_docs": docs}

    def tavily_search(self, query, limit=None):
        api_key = TavilyConfig.api_key
        if not api_key:
            raise ValueError("Tavily API key 未配置")
        max_results = limit or TavilyConfig.max_results
        t0 = time.time()
        resp = httpx.post(
            TavilyConfig.base_url.rstrip("/") + "/search",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "search_depth": TavilyConfig.search_depth,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "include_usage": True,
            },
            timeout=TavilyConfig.timeout,
        )
        elapsed = time.time() - t0
        if resp.status_code != 200:
            # 只记状态码，不记响应体/key；额度耗尽(402/429)同样当作失败，不重试。
            logger.warning(
                f"【{self.name}】Tavily 搜索 HTTP 失败: status={resp.status_code} elapsed={elapsed:.2f}s"
            )
            return []
        data = resp.json()
        results = data.get("results") or []
        usage = data.get("usage")
        logger.info(
            f"【{self.name}】Tavily 搜索完成: status=200 results={len(results)} "
            f"elapsed={elapsed:.2f}s usage={usage}"
        )
        docs = []
        for item in results:
            if not isinstance(item, dict):
                continue
            content = (item.get("content") or item.get("snippet") or "").strip()
            title = item.get("title") or ""
            url = item.get("url") or ""
            if not title and not url and not content:
                continue
            docs.append({"title": title, "content": content, "url": url, "source": "web"})
        return docs


if __name__ == "__main__":
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp.process({"rewritten_query": "三诺生物2022年销售人员"})
    logger.info(json_format(result))
