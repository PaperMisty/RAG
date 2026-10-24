from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(override=True)  # override=True强制用 .env 文件中的值覆盖掉系统环境变量的值。
# load_dotenv 从当前目录开始，递归向上查找.env文件。最先找到的那个.env文件会被加载。

ROOT_DIR = Path(__file__).parents[2]
DATA_DIR = ROOT_DIR / "atguigu" / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"
TEST_DIR = ROOT_DIR / "test"
STATIC_DIR = ROOT_DIR / "web" / "page"


class MineruConfig:
    mineru_token = os.getenv("mineru_token")
    mineru_base_url = os.getenv("mineru_base_url")
    # 解析后端开关：mineru_cloud（默认，现有云端 API）| colab（Colab 网关 /parse）
    parse_backend = os.getenv("PARSE_BACKEND", "mineru_cloud")


class LLMConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    default_model = os.getenv("LLM_DEFAULT_MODEL")
    temperature = float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
    vlm_model = os.getenv("VL_MODEL")
    item_model = os.getenv("ITEM_MODEL")
    model_provider = "openai"


class TextLLMConfig:
    # 【Colab远程部署改造】文本大模型专用配置，与 LLMConfig 分开，原因：
    # LLMConfig 的 base_url / api_key 还被 VL 模型（import_process/nodes/node_md_img.py）
    # 与 Web 搜索 MCP（query_process/nodes/node_web_search_mcp.py 的 token）共用，
    # 不能整体改指向网关；只有文本模型改走 Colab 上的 LiteLLM 网关，用这一组 TEXT_LLM_* 变量。
    api_key = os.getenv("TEXT_LLM_OPENAI_API_KEY")
    base_url = os.getenv("TEXT_LLM_OPENAI_API_BASE")
    default_model = os.getenv("TEXT_LLM_DEFAULT_MODEL")
    model_provider = "openai"
    # 关闭思维链：Qwen3.8 默认在 content 里返回思考过程（以 /think 结束标签收尾），
    # 会污染 RAG 回答正文并大量消耗输出 token；经 ChatOpenAI 的 extra_body 透传给 vLLM。
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}


class MinIOConfig:
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_access = os.getenv("MINIO_ACCESS")
    minio_secret = os.getenv("MINIO_SECRET")
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
    minio_img_dir = os.getenv("MINIO_IMG_DIR")


class EmbeddingConfig:
    bge_m3_path = os.getenv("BGE_M3_PATH")
    bge_m3 = os.getenv("BGE_M3")
    bge_device = os.getenv("BGE_DEVICE")
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16 = os.getenv("BGE_FP16") in ("1", "True", "true", 1)
    # 【Colab远程部署改造】远程 BGE-M3 服务地址（TEI 的 /embed_all 或 FastAPI 回退的 /v1/embeddings）
    bge_m3_url = os.getenv("BGE_M3_URL")
    # 远程向量/重排后端开关：tei（默认）或 fastapi（回退），embedding 与 rerank 共用
    embed_backend = os.getenv("EMBED_BACKEND", "tei")
    # 向量来源开关：remote（默认，现行 HTTP 远程版）| local（本地模型）| colab（Colab 网关）
    embed_source = os.getenv("EMBED_SOURCE", "remote")


class ColabConfig:
    # Colab 推理栈聚合网关（FastAPI + Cloudflare 隧道）公网地址，如 https://xxx.trycloudflare.com
    # 隧道重启后地址会变，只改 .env 不改代码
    colab_base_url = os.getenv("COLAB_BASE_URL", "")
    colab_timeout = float(os.getenv("COLAB_TIMEOUT", "120"))
    # PDF 解析是长任务（数百页年报），单独放宽超时（秒）
    colab_parse_timeout = float(os.getenv("COLAB_PARSE_TIMEOUT", "1800"))


class MilvusConfig:
    milvus_url = os.getenv("MILVUS_URL")
    chunks_collection = os.getenv("CHUNKS_COLLECTION")
    item_name_collection = os.getenv("ITEM_NAME_COLLECTION")


class MongoConfig:
    mongo_url = os.getenv("mongo_url")
    mongo_db_name = os.getenv("mongo_db_name")


class MCPConfig:
    mcp_dashscope_base_url = os.getenv("MCP_DASHSCOPE_BASE_URL")


class TavilyConfig:
    # Tavily Search API. Key is read from env only (.env tavily_key, or
    # standard TAVILY_API_KEY); it must never be logged or hard-coded.
    api_key = os.getenv("tavily_key") or os.getenv("TAVILY_API_KEY")
    base_url = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")
    search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5") or "5")
    timeout = float(os.getenv("TAVILY_TIMEOUT", "15") or "15")


class ReRankerConfig:
    reranker_base_url = os.getenv("RERANK_BASE_URL")
    # 【Colab远程部署改造】自托管 Rerank 服务地址（TEI 的 /rerank 或 FastAPI 回退的 /v1/rerank），
    # 替代原百炼 DashScope 云重排（reranker_tool.py 保留未删，便于回滚）
    reranker_url = os.getenv("BGE_RERANKER_URL")
    # 与 EmbeddingConfig 共用同一个后端开关 EMBED_BACKEND
    embed_backend = os.getenv("EMBED_BACKEND", "tei")


class PromptConfig:
    # 文件命名主体识别
    ITEM_NAME_USER_PROMPT_TEMPLATE = """
                请从以下信息中识别出商品名称与型号：
                文件名：{file_title}

                正文切片（用于辅助识别）：
                {context}

                要求：
                1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
                2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                3. 如果无法识别商品名称,请返回空字符串。
                """
    # Query意图识别
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。你在提取问题的主体名称时，能够区分主体的名称、品牌、型号或者类别。"

    ITEM_NAME_EXTRACT_TEMPLATE = """
    历史会话：
    {history_text}

    当前用户问题：
    {original_query}

    请根据历史会话和当前问题，提取用户正在询问的主体信息。

    重要概念区分：
    - 主体名称：指用户的咨询对象，可以是具体的产品名称（如"烫金机"、"手机"、"笔记本电脑"），也可以是产品的类别（如"打印机"、"吸尘器"、"电磁炉"）。
    - 型号：指商品的具体型号编号，如"hak180"、"iPhone 15"、"X1 Carbon"。
    - 品牌：指商品的品牌，如"华为"、"苹果"、"联想"。
    注意：型号编号和品牌都不是商品名称。

    提取规则：
    1. 优先从历史会话中查找用户最初提到的主体名称（如"烫金机"），型号（如"hak180"）不是主体名称。
    2. 如果历史会话中同时出现了主体名称和型号，item_names 中必须包含主体名称。可以在主体名称前附带品牌 and/or 型号作为补充信息，格式为"[品牌][型号]主体名称"，例如"hak180烫金机"或"华为Mate60手机"。
    3. 如果用户使用了代词（如"这个"、"它"），请结合历史会话指代消解，确定主体名称。
    4. 如果整个对话中确实没有任何主体名称信息，item_names 返回原始语句某个名词。
    5. 请重新改写用户的问题（rewritten_query），使其成为包含主体名称的独立完整问题。
    6. 可能有一个或多个主体，但不能重复。
    """

    HYDE_PROMPT = """
    请基于以下用户查询生成一个简洁的回答范文。
    用户查询: {rewritten_query}
    要求：
    1. 回答要简洁明了，包含核心信息即可
    2. 假设你是该领域的专家，提供专业的解释
    3. 不要使用"假设"、"可能"等不确定的词汇
    4. 保持回答与查询主题高度相关
    5. 使用中文回答且不超过300字
    """

    # 回答生成提示词
    ANSWER_PROMPT = """你是一个智能助手，请根据参考内容回答用户的问题。
    要求：
    1 尽量基于【参考内容】和【用户问题】作答，不要编造不存在的事实。
    2 如果参考内容中包含相关的图片链接（来自 MinIO 的 URL），且回答需要图片辅助说明（例如：外观、结构、接线等），请务必在回答中直接引用这些图片链接。
    3 引用图片时，请严格在相关的文字段落下方另起一行，以如下格式输出图片链接以下是示例，不要照抄内容：

    北京故宫是中国明清两代的皇家宫殿，旧称紫禁城。
    https://example.com/gugong.png

    长城是中国古代的军事防御工程，总长度超过2.1万公里。
    https://example.com/changcheng.png
    
    （注意：图片 URL 必须完全来自于【参考内容】中真实的图片链接，绝对不要编造！如果没有合适图片，则不要输出任何图片链接，也不要输出【图片】字样）

    4 引用标注：当你的回答参考了【参考内容】中的某一部分信息时，请务必在相关的陈述句末尾使用特殊格式标记对应的来源序列 ID。格式为：[[1]]，[[2]] 等。请确保 ID 与【参考内容】中每项前面的序列号（如[1]、[2]）完全一致，绝对不要自己编造序列号！例如：烫金机工作气压需要保持在0.4-0.6MPa [[1]]。（注意：图片引用的优先级最高！如果被参考的内容中包含真实的图片链接，你必须严格在相关的文字段落下方单独起一行直接引用并输出图片 URL，绝对不能因为标注了 [[idx]] 角标而省略输出图片链接！）

    【参考内容】
    {context}

    【历史对话】
    {history}

    【相关商品/实体】
    {item_names}

    【用户问题】
    {question}

    请回答："""

    # 低置信度网络搜索总结提示词
    LOW_CONFIDENCE_ANSWER_PROMPT = """你是一个智能助手。本地数据库未检索到用户问题的相关资料，但我们通过网络搜索找到了一些参考资料。
    请根据【网络搜索资料】和【用户问题】进行整理，回答用户的问题。
    
    【强制要求】
    1. 你的回答开头必须第一句为：“数据库未检索到相关内容，以下是网络搜索结果：\n\n”，不得有任何修改或遗漏！
    2. 基于【网络搜索资料】作答，回答力求准确、客观、简洁，不超过300字。
    3. 使用中文回答。
    4. 引用标注：当你的回答参考了【网络搜索资料】中的某一部分信息时，请务必在相关的陈述句末尾使用特殊格式标记对应的来源序列 ID。格式为：[[1]]，[[2]] 等。请确保 ID 与【网络搜索资料】中每项前面的序列号完全一致，绝对不要自己编造序列号！例如：特朗普出生于纽约 [[1]]。

    【网络搜索资料】
    {context}

    【历史对话】
    {history}

    【用户问题】
    {question}

    请回答："""
