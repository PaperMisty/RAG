# atguigu/query_process/nodes/node_item_name_confirm.py
from bson import ObjectId
from atguigu.tool.mongo_client_tool import update_item_name_and_query
from atguigu.tool.mongo_client_tool import clear_history
from atguigu.config.config import MilvusConfig
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.milvus_client_tool import weighted_hybrid_search
from atguigu.tool.milvus_client_tool import create_reqs
# 【Colab远程部署改造】原本地模型版 bgem3_client_tool 改为远程 HTTP 版（函数签名与返回结构不变）
from atguigu.tool.bgem3_tool import get_bgem3_embedding  # 改走统一分发(EMBED_SOURCE),与检索/入库一致;原写死远程 HTTP 版
from atguigu.config.config import LLMConfig, PromptConfig, TextLLMConfig  # 【Colab远程部署改造】文本模型走网关配置
from langchain.chat_models import init_chat_model
from atguigu.tool.mongo_client_tool import get_recent_history_list
from atguigu.tool.mongo_client_tool import add_or_update_data
import json, time
from rich import print
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger
from pydantic import BaseModel, Field


class ItemNameExtractResult(BaseModel):
    """大模型意图提取和改写结果结构"""

    rewritten_query: str = Field(
        default="", description="改写后的用户查询，需清晰阐述出本次对话的核心咨询意图和具体商品型号"
    )
    item_names: list[str] = Field(
        default_factory=list,
        description="从当前用户提问和整个历史会话上下文中，挖掘出当前正在讨论的主体商品/设备名称",
    )


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")
        # 1. 获取基本对话信息
        context, original_query, message_id, session_id = self.get_history_context(state)
        # 2. 进行意图识别,重写问题
        rewritten_query, item_names = self.get_rewritten_and_item_names(context, original_query)
        # 3. 进行向量匹配,对齐主体信息,并生成助手回复
        rewritten_query, answer, final_item_lst, message_id, is_low_confidence = self.get_hierarchical_align_item_name(
            session_id, message_id, rewritten_query, original_query, item_names
        )

        return {
            "message_id": message_id,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "item_names": final_item_lst,
            "answer": answer,
            "is_low_confidence": is_low_confidence,
        }

    def get_history_context(self, state: QueryGraphState) -> tuple[str, str, ObjectId, str]:
        """获取历史会话

        Args:
            state (QueryGraphState):

        Raises:
            ValueError:
            ValueError:

        Returns:
            tuple: (context, original_query, message_id, session_id)
        """
        # 取对话id
        session_id = state.get("session_id", "")
        if not session_id:
            logger.error("session_id 为空")
            raise ValueError("session_id 为空")

        # 取用户的问题
        original_query = state.get("original_query", "")
        if not original_query:
            logger.error("用户问题为空")
            raise ValueError("用户问题为空")

        # 最新对话写入mongodb
        message_id = add_or_update_data(
            {
                "session_id": session_id,
                "role": "user",
                "text": original_query,
                "rewritten_query": None,
                "item_names": None,
                "ts": time.time(),
            }
        )
        logger.info(f"最新对话写入mongodb,message_id: {message_id}")

        # 取最近历史对话记录
        recent_history_list = get_recent_history_list(session_id)
        if not recent_history_list:
            logger.warning("最近历史对话记录为空")
            return "", original_query, message_id, session_id

        # 将最近历史对话记录格式化为字符串
        context = "\n".join([f"{item['role']}: {item['text']}" for item in recent_history_list])
        return context, original_query, message_id, session_id

    def get_rewritten_and_item_names(self, context, original_query) -> tuple[str, list]:
        """获取改写后的问题和商品名称

        Args:
            context (str): 历史会话
            original_query (str): 用户原始问题

        Returns:
            tuple: (rewritten_query, item_names)
        """
        llm = init_chat_model(
            model=TextLLMConfig.default_model,
            model_provider=TextLLMConfig.model_provider,
            base_url=TextLLMConfig.base_url,
            api_key=TextLLMConfig.api_key,
            extra_body=TextLLMConfig.extra_body,
            temperature=LLMConfig.temperature,
        )

        # 绑定结构化输出 schema
        structured_llm = llm.with_structured_output(schema=ItemNameExtractResult, include_raw=True)

        # 让LLM总结rewritten_query字符 和 item_names列表
        msg = [
            {"role": "system", "content": PromptConfig.ITEM_NAME_EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PromptConfig.ITEM_NAME_EXTRACT_TEMPLATE.format(
                    history_text=context, original_query=original_query
                ),
            },
        ]

        response = structured_llm.invoke(msg)

        # 当 include_raw=True 开启时，打印原始 LLM 输出以方便排查
        parsed_response = None
        if isinstance(response, dict) and "raw" in response:
            raw_message = response["raw"]
            logger.info("=" * 60)
            logger.info(f"大模型原始输出 (raw) 类型: {type(raw_message)}")
            if hasattr(raw_message, "content"):
                logger.info(f"大模型原始消息正文 (raw.content): \n{raw_message.content}")
            if hasattr(raw_message, "additional_kwargs"):
                logger.info(f"大模型原始附加参数 (raw.additional_kwargs): \n{raw_message.additional_kwargs}")
            if hasattr(raw_message, "tool_calls"):
                logger.info(f"大模型原始工具调用 (raw.tool_calls): \n{raw_message.tool_calls}")
            logger.info("=" * 60)

            # 提取解析后的结构
            parsed_response = response.get("parsed")
        else:
            parsed_response = response

        # 解析返回结果，拦截大模型返回 None 的特殊边界情况并进行多类型（dict 与 Pydantic）兼容解析
        if parsed_response is None:
            logger.warning("大模型返回的解析结果为空 (None)")
            rewritten_query = original_query
            item_names = []
        elif isinstance(parsed_response, dict):
            rewritten_query = parsed_response.get("rewritten_query", "")
            item_names = parsed_response.get("item_names", [])
            logger.info(f"意图识别成功(Dict),改写后: {rewritten_query},提取主体: {item_names}")
        elif isinstance(parsed_response, ItemNameExtractResult):
            rewritten_query = parsed_response.rewritten_query
            item_names = parsed_response.item_names
            logger.info(f"意图识别成功(Pydantic),改写后: {rewritten_query},提取主体: {item_names}")
        else:
            logger.warning(f"大模型返回了非预期的对象类型: {type(parsed_response)}")
            rewritten_query = getattr(parsed_response, "rewritten_query", original_query)
            item_names = getattr(parsed_response, "item_names", [])

        # 强力清洗逻辑：过滤大模型可能幻觉返回的空字符串或纯空格项
        if isinstance(item_names, list):
            item_names = [name.strip() for name in item_names if name and isinstance(name, str) and name.strip()]

        # 强力防御逻辑：如果模型强行填充了 None、空字符串等，在此做归一化
        if not rewritten_query:
            rewritten_query = original_query
        if not item_names:
            item_names = [rewritten_query]
        elif isinstance(item_names, str):
            item_names = [item_names]

        return rewritten_query, item_names

    def get_hierarchical_align_item_name(self, session_id, message_id, rewritten_query, original_query, item_names):
        """对检索到的商品名称进行向量匹配和层级对齐，最终生成回复并更新历史会话

        Args:
            session_id (str): 会话 ID
            message_id (ObjectId): 提问的消息 ID
            rewritten_query (str): 重写后的意图问题
            original_query (str): 原始的用户问题
            item_names (list[str]): 意图识别提取到的商品名称列表

        Returns:
            tuple: (rewritten_query, answer, final_item_lst, message_id)
        """
        final_item_lst = []
        answer = ""

        # - 防御没有提取到意图重写内容
        if not rewritten_query:
            logger.warning("未提取到意图重写内容，直接使用原查询")
            rewritten_query = original_query

        # - 防御没有提取到商品名称
        is_low_confidence = False
        if not item_names:
            logger.warning("未提取到商品名称")
            is_low_confidence = True
        else:
            # 1. 向量数据库混合检索物料
            matched_results = self.search_item_names_in_milvus(item_names)
            # 2. 对分值进行层级化评估对齐
            final_item_lst, answer, is_low_confidence = self.align_item_names_by_score(matched_results)
            # 3. 完成 MongoDB 会话状态回写
            message_id = self.handle_session_history_update(
                session_id, message_id, rewritten_query, final_item_lst, answer
            )

        return rewritten_query, answer, final_item_lst, message_id, is_low_confidence

    def search_item_names_in_milvus(self, item_names: list[str]) -> list[dict]:
        """批量对LLM提取的假设商品名称进行向量检索，提取检索结果的真实商品名及对应相似度分值

        Args:
            item_names (list[str]): 待检索的商品名称列表

        Returns:
            list[dict]: 混合检索候选物料列表
        """
        # 去除空白字符
        cleaned_item_names = [item.replace(" ", "").replace("\n", "").replace("\t", "") for item in item_names]

        # 获取稀疏/稠密双路向量
        vecs = get_bgem3_embedding(cleaned_item_names)
        matched_results = []

        for idx, item_name in enumerate(cleaned_item_names):
            reqs = create_reqs(
                dense_data=vecs["dense"][idx],
                sparse_data=vecs["sparse"][idx],
                dense_ann_field="dense",
                sparse_ann_field="sparse",
                limit=10,
            )
            result = weighted_hybrid_search(
                collection_name=MilvusConfig.item_name_collection,
                reqs=reqs,
                ranker=[0.8, 0.2],
                limit=10,
                output_fields=["item_name"],
            )
            # ==========测试检索结果是否含有 '空字符串' ========
            if result:
                logger.debug(f"==={result[0]=}===")
                matched_results.extend(result[0])

        return matched_results

    def align_item_names_by_score(self, matched_results: list[dict]) -> tuple[list[str], str, bool]:
        """根据相似度分值将检索到的物料归入高置信度（确认）或中置信度（候选）分类中

        Args:
            matched_results (list[dict]): 向量检索得到的物料候选列表

        Returns:
            tuple: (final_item_lst, answer)
        """
        choosed_item_lst = []
        optional_item_lst = []

        for item_info in matched_results:
            item_name_val = item_info.get("entity", {}).get("item_name")
            if not item_name_val or item_name_val.strip() in ["", "空字符串", "None", "null"]:
                continue

            distance = item_info.get("distance", 0.0)
            if distance >= 0.8:
                # 高置信度（可以直接确认）
                if item_name_val not in choosed_item_lst:
                    choosed_item_lst.append(item_name_val)
                logger.info(f"高置信度匹配: {item_name_val}")
            elif 0.6 < distance < 0.8:
                # 中等置信度（需提供给用户做澄清候选）
                if item_name_val not in optional_item_lst:
                    optional_item_lst.append(item_name_val)
                logger.info(f"中等置信度匹配: {item_name_val}")

        final_item_lst = []
        answer = ""
        is_low_confidence = False

        logger.info(f"choosed_item_lst: {choosed_item_lst}, optional_item_lst: {optional_item_lst}")
        if choosed_item_lst:
            # 存在高置信度匹配，直接对齐使用
            final_item_lst = choosed_item_lst
        elif len(optional_item_lst) == 1:
            # 中置信度有一个,直接对齐使用
            final_item_lst = optional_item_lst
        elif len(optional_item_lst) > 1:
            # 中置信度有多个，拼装交互澄清话术
            tmp = "\t".join(optional_item_lst)
            answer = f"您想咨询的是以下哪一个? \n: {tmp} "
        else:
            # 置信度过低或数据库无该内容，此处不再直接生成拒答 answer，而是标记低置信度，交给网络搜索
            is_low_confidence = True
            logger.info("置信度过低或无匹配主体，标记 is_low_confidence = True，后续交由网络搜索")

        return final_item_lst, answer, is_low_confidence

    def handle_session_history_update(
        self, session_id: str, message_id: ObjectId, rewritten_query: str, final_item_lst: list[str], answer: str
    ) -> ObjectId:
        """完成 MongoDB 对话记录的相关写入与回填

        Args:
            session_id (str): 会话 ID
            message_id (ObjectId): 原始提问的消息 ID
            rewritten_query (str): 重写后的问题
            final_item_lst (list[str]): 已确认对齐的商品列表
            answer (str): 助手的回复文本

        Returns:
            ObjectId: 最新操作的消息记录 ID
        """
        # 如果需要助手做出确认/警告回复，将回复内容新增到对话历史中，并更新 message_id
        if answer:
            data_dict = {
                "session_id": session_id,
                "role": "assistant",
                "text": answer,
                "rewritten_query": rewritten_query,
                "item_names": final_item_lst,
                "ts": time.time(),
            }
            message_id = add_or_update_data(data_dict)
            logger.info(f"助手回复写入mongodb,message_id: {message_id}")

        # 无论置信度高低，都对最近的历史记录回写 rewritten_query 与已对齐的商品名称列表
        # 避免情况: 用户先问A, 查询到相关主体后回写了; 后问B, 没有查询到主体, 导致最新主体信息没有更新
        update_item_name_and_query(session_id, rewritten_query, final_item_lst)
        logger.info(f"回写主体信息: rewritten_query: {rewritten_query}, item_names: {final_item_lst}")

        return message_id


if __name__ == "__main__":
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

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # logger.info(json_format(result))
    print(result)
