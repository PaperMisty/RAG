# atguigu/query_process/nodes/node_answer_output.py

from atguigu.tool.json_format_tool import json_format
from atguigu.tool.mongo_client_tool import add_or_update_data
from atguigu.config.config import LLMConfig, TextLLMConfig  # 【Colab远程部署改造】文本模型走网关配置
from langchain.chat_models import init_chat_model
from atguigu.tool.mongo_client_tool import get_recent_history_list
from atguigu.config.config import PromptConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
import time


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
            q.put({"event": "delta", "data": {"delta": answer}})
            q.put({"event": "final", "data": ""})
            # 对话写入mongodb
            # insert_id = add_or_update_data(data_dict) 已经在主体识别Node中写入了
            # logger.info(f"answer输出信息已插入数据库, id:{insert_id}")
        # 置信度极低或主体未识别，网络搜索总结情况
        elif is_low_confidence:
            web_search_docs = state.get("web_search_docs", [])
            if not web_search_docs:
                # Tavily 失败/空结果且低置信：不进 LLM，直接回固定句，避免模型编造“网络搜索结果”。
                answer = "[\u672a\u68c0\u7d22\u5230\u76f8\u5173\u4e3b\u4f53]"
                q.put({"event": "chunks", "data": []})
                q.put({"event": "delta", "data": {"delta": answer}})
                q.put({"event": "final", "data": ""})
                data_dict["text"] = answer
                data_dict["chunks"] = []
                insert_id = add_or_update_data(data_dict)
                logger.info(f"低置信度且网络搜索无结果，直接返回固定句，未调用 LLM, id:{insert_id}")
                return {"answer": answer}
            combine_content = ""
            for idx, doc in enumerate(web_search_docs, 1):
                content = f'[{idx}][来源: {doc.get("title")}][网址: {doc.get("url")}]\n内容摘要: {doc.get("content")}\n\n'
                combine_content += content
                
            # 拼接历史对话记录
            history_session = get_recent_history_list(session_id, limit=3)
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
            q.put({"event": "chunks", "data": web_chunks})

            # 拿到流式对话的生成器对象, 放入queue
            res_generator = llm.stream(msg)
            answer = ""
            for res_delta in res_generator:
                q.put({"event": "delta", "data": {"delta": res_delta.content}})
                answer += res_delta.content
            q.put({"event": "final", "data": ""})
            data_dict["text"] = answer
            data_dict["chunks"] = web_chunks
            insert_id = add_or_update_data(data_dict)
            logger.info(f"低置信度网络搜索总结信息已插入数据库, id:{insert_id}")
        # 置信度较高情况, 触发常规本地RAG回答
        else:
            chunk_dict_list = state.get("reranked_docs")
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
                print(matches)
            else:
                print("不存在图片!!!")
                logger.info(json_format(chunk_dict_list[:3]))

            # ===================
            # 拼接历史对话记录
            history_session = get_recent_history_list(session_id, limit=10)
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
            q.put({"event": "chunks", "data": local_chunks})

            # 拿到流式对话的生成器对象, 放入queue
            res_generator = llm.stream(msg)
            for res_delta in res_generator:
                q.put({"event": "delta", "data": {"delta": res_delta.content}})
                answer += res_delta.content
            q.put({"event": "final", "data": ""})
            data_dict["text"] = answer
            data_dict["chunks"] = local_chunks
            insert_id = add_or_update_data(data_dict)
            logger.info(f"answer输出信息已插入数据库, id:{insert_id}")
        return {"answer": answer}
