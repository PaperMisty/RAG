# atguigu/query_process/nodes/node_answer_output.py

from atguigu.tool.mongo_client_tool import add_or_update_data
from atguigu.config.config import LLMConfig, TextLLMConfig  # 【Colab远程部署改造】文本模型走网关配置
from langchain.chat_models import init_chat_model
from atguigu.tool.mongo_client_tool import get_recent_history_list
from atguigu.config.config import PromptConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger, summarize_docs
from observer.collector import trace_collector
import queue as queue_module
import threading
import time


# LLM 流式生成的超时底线。init_chat_model / llm.stream 本身没有
# timeout 参数，实测可无限挂起（前端表现为一直转圈）。120 秒无新
# 内容或总计 300 秒即判超时，走统一的 error 事件链路。注意：超时
# 后生成线程仍在后台跑完（daemon 线程，无法安全杀掉），只是图
# 不再等它，前端不再静默挂起。
_STREAM_STALL_TIMEOUT_S = 120
_STREAM_OVERALL_TIMEOUT_S = 300


def _iter_stream_with_timeout(llm, msg):
    """Yield llm.stream deltas, raising TimeoutError on stall/overall timeout."""
    result_queue = queue_module.Queue()

    def _run():
        try:
            for res_delta in llm.stream(msg):
                result_queue.put(("delta", res_delta.content))
            result_queue.put(("done", None))
        except Exception as exc:  # noqa: BLE001 - re-raised to the caller
            result_queue.put(("error", exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started = time.time()
    last_progress = started
    while True:
        wait = min(
            _STREAM_OVERALL_TIMEOUT_S - (time.time() - started),
            _STREAM_STALL_TIMEOUT_S - (time.time() - last_progress),
        )
        if wait <= 0:
            raise TimeoutError("答案生成超时（流式生成停滞或总耗时超限），请重试")
        try:
            kind, payload = result_queue.get(timeout=wait)
        except queue_module.Empty:
            raise TimeoutError("答案生成超时（120 秒内无新内容），请重试")
        last_progress = time.time()
        if kind == "done":
            return
        if kind == "error":
            raise payload
        yield payload


class NodeAnswerOutput(NodeBase):
    """
    节点功能: 答案生成
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_answer_output"

    def process(self, state: QueryGraphState):
        """
        将查询数据库得到的信息给到LLM,作为answer回答, 若置信度较低,则直接返回answer
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        task_id = state.get("task_id") or ""
        t_out_start = time.perf_counter()
        item_names = state.get("item_names")
        rewritten_query = state.get("rewritten_query")
        q = state.get("q")
        answer = state.get("answer")
        session_id = state.get("session_id")
        is_low_confidence = state.get("is_low_confidence", False)
        # 构造mongodb对话结构
        data_dict = {
            "session_id": session_id,
            "task_id": state.get("task_id"),
            "role": "assistant",
            "text": answer,
            "rewritten_query": rewritten_query,
            "item_names": item_names,
            "ts": time.time(),
        }
        # 置信度较低/中等情况, 触发直接回答（中置信度多个选项的确认澄清提示）
        if answer:
            if q is not None:
                q.put({"event": "delta", "data": {"delta": answer}})
                q.put({"event": "final", "data": ""})
            out_ms = round((time.perf_counter() - t_out_start) * 1000, 2)
            if task_id:
                trace_collector.add_io_item(task_id, "output", "澄清选项直接下发 (无LLM调用)", out_ms, io_type="cpu")
                trace_collector.set_module_total_ms(task_id, "output", out_ms)
                trace_collector.record_output(
                    task_id=task_id,
                    final_answer=answer,
                    output_mode="clarify_direct",
                    model_name="Rule-Clarify",
                    context_chunk_count=0,
                    ttft_ms=out_ms,
                )
        # 置信度极低或主体未识别，网络搜索总结情况
        elif is_low_confidence:
            web_search_docs = state.get("web_search_docs", [])
            if not web_search_docs:
                # Tavily 失败/空结果且低置信：不进 LLM，直接回固定句，避免模型编造“网络搜索结果”。
                answer = "[\u672a\u68c0\u7d22\u5230\u76f8\u5173\u4e3b\u4f53]"
                if q is not None:
                    q.put({"event": "chunks", "data": []})
                    q.put({"event": "delta", "data": {"delta": answer}})
                    q.put({"event": "final", "data": ""})
                data_dict["text"] = answer
                data_dict["chunks"] = []
                t_mongo = time.perf_counter()
                insert_id = add_or_update_data(data_dict)
                mongo_ms = round((time.perf_counter() - t_mongo) * 1000, 2)
                out_ms = round((time.perf_counter() - t_out_start) * 1000, 2)
                if task_id:
                    trace_collector.add_io_item(task_id, "output", "MongoDB 落库 IO", mongo_ms, io_type="db")
                    trace_collector.set_module_total_ms(task_id, "output", out_ms)
                    trace_collector.record_output(
                        task_id=task_id,
                        final_answer=answer,
                        output_mode="low_conf_fixed",
                        model_name="Fixed-Fallback",
                        context_chunk_count=0,
                        ttft_ms=0.0,
                    )
                logger.info(f"低置信度且网络搜索无结果，直接返回固定句，未调用 LLM, id:{insert_id}")
                return {"answer": answer}
            combine_content = ""
            for idx, doc in enumerate(web_search_docs, 1):
                content = f'[{idx}][来源: {doc.get("title")}][网址: {doc.get("url")}]\n内容摘要: {doc.get("content")}\n\n'
                combine_content += content
                
            # 拼接历史对话记录
            t_hist = time.perf_counter()
            history_session = get_recent_history_list(session_id, limit=3)
            hist_ms = round((time.perf_counter() - t_hist) * 1000, 2)
            combine_session = ""
            for session in history_session:
                content = f'[{session.get("role")}:][{session.get("text")}]'
                combine_session += content

            # 构造提示词给LLM
            prompt = PromptConfig.LOW_CONFIDENCE_ANSWER_PROMPT.format(
                context=combine_content, history=combine_session, question=rewritten_query
            )
            llm = init_chat_model(
                model=TextLLMConfig.default_model,
                model_provider=TextLLMConfig.model_provider,
                base_url=TextLLMConfig.base_url,
                api_key=TextLLMConfig.api_key,
            extra_body=TextLLMConfig.extra_body,
                temperature=LLMConfig.temperature,
            )
            msg = [{"role": "user", "content": prompt}]
            # 新增：向前端推送网络搜索对应的 chunks 数据包以便定位展示
            web_chunks = [
                {
                    "id": str(idx),
                    "file_title": doc.get("title", "网络搜索结果"),
                    "section_title": doc.get("url", ""),
                    "content": doc.get("content", "")
                }
                for idx, doc in enumerate(web_search_docs, 1)
            ]
            if q is not None:
                q.put({"event": "chunks", "data": web_chunks})

            # 拿到流式对话的生成器对象, 放入queue（带停滞超时，见文件头 helper）
            answer = ""
            t_llm_start = time.perf_counter()
            ttft_ms = 0.0
            first_token = True
            for delta in _iter_stream_with_timeout(llm, msg):
                if first_token and delta:
                    ttft_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
                    first_token = False
                if q is not None:
                    q.put({"event": "delta", "data": {"delta": delta}})
                answer += delta
            llm_stream_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
            if q is not None:
                q.put({"event": "final", "data": ""})
            data_dict["text"] = answer
            data_dict["chunks"] = web_chunks
            t_mongo = time.perf_counter()
            insert_id = add_or_update_data(data_dict)
            mongo_ms = round((time.perf_counter() - t_mongo) * 1000, 2)
            out_ms = round((time.perf_counter() - t_out_start) * 1000, 2)
            if task_id:
                trace_collector.add_io_item(task_id, "output", "MongoDB 读取历史 IO", hist_ms, io_type="db")
                trace_collector.add_io_item(task_id, "output", "LLM 首字延迟 (TTFT)", ttft_ms, io_type="llm")
                trace_collector.add_io_item(task_id, "output", "LLM 流式生成总 IO", llm_stream_ms, io_type="llm")
                trace_collector.add_io_item(task_id, "output", "MongoDB 结果落库 IO", mongo_ms, io_type="db")
                trace_collector.set_module_total_ms(task_id, "output", out_ms)
                trace_collector.record_output(
                    task_id=task_id,
                    final_answer=answer,
                    output_mode="low_conf_web_llm",
                    model_name=str(TextLLMConfig.default_model or "LLM"),
                    context_chunk_count=len(web_search_docs),
                    ttft_ms=ttft_ms,
                )
            logger.info(f"低置信度网络搜索总结信息已插入数据库, id:{insert_id}")
        # 置信度较高情况, 触发常规本地RAG回答
        else:
            chunk_dict_list = state.get("reranked_docs") or []
            # 拼接chunk_dict内容
            combine_content = ""
            for idx, chunk_dict in enumerate(chunk_dict_list, 1):
                content = f'[{idx}][{chunk_dict["source"]}][{chunk_dict["title"]}]\n[{chunk_dict["content"]}]\n\n'
                combine_content += content
            # ===================
            # 测试,查看chunk_content是否存在链接
            import re

            md_img_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
            matches = md_img_pattern.findall(combine_content)
            if matches:
                logger.info(f"chunk 内容含图片链接 {len(matches)} 条")
            else:
                logger.info("chunk 内容无图片链接")
                logger.info(f"高置信召回 chunk: {summarize_docs(chunk_dict_list, limit=3)}")

            # ===================
            # 拼接历史对话记录
            t_hist = time.perf_counter()
            history_session = get_recent_history_list(session_id, limit=10)
            hist_ms = round((time.perf_counter() - t_hist) * 1000, 2)
            combine_session = ""
            for session in history_session:
                content = f'[{session.get("role")}:][{session.get("text")}]'
                combine_session += content

            # 构造提示词给LLM
            prompt = PromptConfig.ANSWER_PROMPT.format(
                context=combine_content, history=combine_session, item_names=item_names, question=rewritten_query
            )
            llm = init_chat_model(
                model=TextLLMConfig.default_model,
                model_provider=TextLLMConfig.model_provider,
                base_url=TextLLMConfig.base_url,
                api_key=TextLLMConfig.api_key,
            extra_body=TextLLMConfig.extra_body,
                temperature=LLMConfig.temperature,
            )
            msg = [{"role": "user", "content": prompt}]
            # 新增：将本地召回的切片数据包以序列号 ID 形式推送给前端，与大模型输出的序列标号对齐
            local_chunks = [
                {
                    "id": str(idx),
                    "file_title": chunk.get("file_title") or chunk.get("source") or "未知文档",
                    "section_title": chunk.get("section_title") or chunk.get("title") or "正文",
                    "content": chunk.get("content", "")
                }
                for idx, chunk in enumerate(chunk_dict_list, 1)
            ]
            if q is not None:
                q.put({"event": "chunks", "data": local_chunks})

            # 拿到流式对话的生成器对象, 放入queue（带停滞超时，见文件头 helper）
            answer = answer or ""
            t_llm_start = time.perf_counter()
            ttft_ms = 0.0
            first_token = True
            for delta in _iter_stream_with_timeout(llm, msg):
                if first_token and delta:
                    ttft_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
                    first_token = False
                if q is not None:
                    q.put({"event": "delta", "data": {"delta": delta}})
                answer += delta
            llm_stream_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
            if q is not None:
                q.put({"event": "final", "data": ""})
            data_dict["text"] = answer
            data_dict["chunks"] = local_chunks
            t_mongo = time.perf_counter()
            insert_id = add_or_update_data(data_dict)
            mongo_ms = round((time.perf_counter() - t_mongo) * 1000, 2)
            out_ms = round((time.perf_counter() - t_out_start) * 1000, 2)
            if task_id:
                trace_collector.add_io_item(task_id, "output", "MongoDB 读取历史 IO", hist_ms, io_type="db")
                trace_collector.add_io_item(task_id, "output", "LLM 首字延迟 (TTFT)", ttft_ms, io_type="llm")
                trace_collector.add_io_item(task_id, "output", "LLM 流式生成总 IO", llm_stream_ms, io_type="llm")
                trace_collector.add_io_item(task_id, "output", "MongoDB 结果落库 IO", mongo_ms, io_type="db")
                trace_collector.set_module_total_ms(task_id, "output", out_ms)
                trace_collector.record_output(
                    task_id=task_id,
                    final_answer=answer,
                    output_mode="rag_llm",
                    model_name=str(TextLLMConfig.default_model or "LLM"),
                    context_chunk_count=len(chunk_dict_list),
                    ttft_ms=ttft_ms,
                )
            logger.info(f"answer输出信息已插入数据库, id:{insert_id}")
        return {"answer": answer}
