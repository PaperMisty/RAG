# observer/schemas.py

from typing import Any
from pydantic import BaseModel, Field


class IOBreakdownItem(BaseModel):
    """职责: 记录单个网络IO或计算子步骤的耗时明细"""

    name: str = Field(..., description="IO操作或子步骤名称")
    io_type: str = Field(default="network", description="类型: network | db | llm | cpu")
    duration_ms: float = Field(default=0.0, description="耗时(毫秒)")
    status: str = Field(default="ok", description="状态: ok | timeout | error")
    detail: str = Field(default="", description="补充说明信息")


class ModuleTiming(BaseModel):
    """职责: 记录单个模块方格的总耗时与网络IO分项耗时"""

    started_at: float = Field(default=0.0, description="开始时间戳(秒)")
    ended_at: float = Field(default=0.0, description="结束时间戳(秒)")
    total_ms: float = Field(default=0.0, description="模块总耗时(毫秒)")
    io_breakdown: list[IOBreakdownItem] = Field(default_factory=list, description="内部IO耗时拆解列表")


class TraceChunkItem(BaseModel):
    """职责: 表示检索、RRF、Reranker、断崖检测各阶段中的单个文档切片项"""

    rank: int = Field(..., description="当前阶段内的排名(从1开始)")
    chunk_id: str = Field(default="", description="切片唯一标识")
    file_title: str = Field(default="", description="来源文档标题")
    section_title: str = Field(default="", description="小节标题")
    item_name: str = Field(default="", description="所属商品/主体名称")
    content: str = Field(default="", description="切片正文内容")
    url: str = Field(default="", description="网络搜索来源URL(如有)")
    source: str = Field(default="local", description="来源标识: dense | sparse | hyde | local | web")
    score: float = Field(default=0.0, description="当前阶段主得分")
    raw_score: float | None = Field(default=None, description="合并前的原始得分")
    dense_rank: int | None = Field(default=None, description="在Dense检索中的排名")
    sparse_rank: int | None = Field(default=None, description="在Sparse检索中的排名")
    hyde_rank: int | None = Field(default=None, description="在HyDE检索中的排名")
    rrf_rank: int | None = Field(default=None, description="在RRF融合阶段的排名")
    is_grayed: bool = Field(default=False, description="是否灰化展示(被挤出TopK或被断崖淘汰)")
    gray_reason: str = Field(default="", description="灰化原因说明")
    abs_gap: float | None = Field(default=None, description="与前一名切片的绝对分差")
    ratio_gap: float | None = Field(default=None, description="与前一名切片的相对分差比例")


class CandidateItemScore(BaseModel):
    """职责: 记录Milvus主体名称向量对齐的候选物料及得分"""

    item_name: str = Field(default="", description="候选主体名称")
    score: float = Field(default=0.0, description="混合检索相似度得分")
    level: str = Field(default="low", description="置信度等级: high | medium | low")


class RouteBranchStatus(BaseModel):
    """职责: 描述单条路由分支线的激活状态与说明"""

    route_key: str = Field(..., description="路由键: high_hybrid | medium_clarify | low_web_only")
    label: str = Field(..., description="路由线展示名称")
    condition: str = Field(default="", description="触发条件描述")
    is_active: bool = Field(default=False, description="是否为当前实际执行的路由线")
    target_desc: str = Field(default="", description="下游目标模块描述")


class QueryModuleTrace(BaseModel):
    """职责: 记录Query理解、改写、关键词与主体提取模块的数据"""

    original_query: str = Field(default="", description="用户原始问题")
    rewritten_query: str = Field(default="", description="LLM改写后的问题")
    key_words: list[str] = Field(default_factory=list, description="提取的核心关键词列表")
    extracted_item_names: list[str] = Field(default_factory=list, description="LLM初步提取的主体名称列表")
    aligned_item_names: list[str] = Field(default_factory=list, description="Milvus对齐确认后的最终主体名称列表")
    history_turns: int = Field(default=0, description="参考的历史对话轮数")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与网络IO")


class RouterModuleTrace(BaseModel):
    """职责: 记录主体置信度分数与三路分支路由状态的数据"""

    confidence_score: float = Field(default=0.0, description="最高主体匹配置信度分数")
    threshold_high: float = Field(default=0.8, description="高置信度阈值")
    threshold_low: float = Field(default=0.6, description="低置信度阈值")
    route_taken: str = Field(default="high_hybrid", description="实际路由: high_hybrid | medium_clarify | low_web_only")
    route_reason: str = Field(default="", description="路由判定原因说明")
    matched_candidates: list[CandidateItemScore] = Field(default_factory=list, description="Milvus主体匹配候选得分列表")
    routes: list[RouteBranchStatus] = Field(default_factory=list, description="三条路由分支线的激活/灰化状态")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与网络IO")


class RetrievalModuleTrace(BaseModel):
    """职责: 记录Dense稠密检索、Sparse稀疏检索及补充检索的排名切片数据"""

    executed: bool = Field(default=False, description="是否执行了向量检索模块")
    dense_chunks: list[TraceChunkItem] = Field(default_factory=list, description="Dense稠密检索按排名切片列表")
    sparse_chunks: list[TraceChunkItem] = Field(default_factory=list, description="Sparse稀疏检索按排名切片列表")
    hybrid_chunks: list[TraceChunkItem] = Field(default_factory=list, description="Re-Query混合检索按排名切片列表")
    hyde_query: str = Field(default="", description="HyDE生成的假设性答案文本")
    hyde_chunks: list[TraceChunkItem] = Field(default_factory=list, description="HyDE检索按排名切片列表")
    web_docs: list[TraceChunkItem] = Field(default_factory=list, description="Tavily网络搜索结果列表")
    dense_timing: ModuleTiming = Field(default_factory=ModuleTiming, description="Dense检索子模块耗时与IO")
    sparse_timing: ModuleTiming = Field(default_factory=ModuleTiming, description="Sparse检索子模块耗时与IO")
    hyde_timing: ModuleTiming = Field(default_factory=ModuleTiming, description="HyDE检索子模块耗时与IO")
    web_timing: ModuleTiming = Field(default_factory=ModuleTiming, description="Web搜索子模块耗时与IO")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="检索阶段整体耗时与IO汇总")


class RrfModuleTrace(BaseModel):
    """职责: 记录RRF倒排融合排序后的TopK与尾部灰化切片数据"""

    executed: bool = Field(default=False, description="是否执行了RRF模块")
    rrf_smooth: int = Field(default=60, description="RRF平滑常数k")
    top_k: int = Field(default=10, description="RRF保留的TopK阈值")
    total_unique_count: int = Field(default=0, description="去重融合后的总切片数")
    kept_count: int = Field(default=0, description="进入TopK的切片数")
    grayed_count: int = Field(default=0, description="被挤在TopK后面标灰的切片数")
    chunks: list[TraceChunkItem] = Field(default_factory=list, description="RRF按排名排列的全部切片(含TopK后标灰项)")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与IO")


class RerankerModuleTrace(BaseModel):
    """职责: 记录Cross-Encoder重排序后的完整切片排名数据"""

    executed: bool = Field(default=False, description="是否执行了Reranker模块")
    model_name: str = Field(default="bge-reranker-v2-m3 / gte-rerank", description="重排模型名称")
    input_count: int = Field(default=0, description="送入重排的文档总数")
    chunks: list[TraceChunkItem] = Field(default_factory=list, description="Reranker重排后的按排名切片列表")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与网络IO")


class CliffModuleTrace(BaseModel):
    """职责: 记录断崖检测参数、截断位置及被断崖灰化的切片数据"""

    executed: bool = Field(default=False, description="是否执行了断崖检测模块")
    abs_gap_threshold: float = Field(default=0.25, description="绝对分差断崖阈值")
    ratio_gap_threshold: float = Field(default=0.10, description="相对分差断崖阈值")
    min_len: int = Field(default=3, description="最少保留切片数")
    max_len: int = Field(default=10, description="最多保留切片数")
    cliff_triggered: bool = Field(default=False, description="是否触发了分差断崖截断")
    cliff_after_rank: int = Field(default=0, description="在第几名之后发生截断")
    trigger_reason: str = Field(default="", description="截断触发原因描述")
    kept_count: int = Field(default=0, description="断崖检测后保留的切片数")
    dropped_count: int = Field(default=0, description="被断崖淘汰标灰的切片数")
    chunks: list[TraceChunkItem] = Field(default_factory=list, description="含分差指标与断崖标灰状态的完整切片列表")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与IO")


class OutputModuleTrace(BaseModel):
    """职责: 记录最终大模型输出文本、引用角标与流式生成耗时"""

    executed: bool = Field(default=False, description="是否执行了输出模块")
    output_mode: str = Field(default="rag_llm", description="输出模式: rag_llm | clarify_direct | low_conf_web_llm | low_conf_fixed")
    model_name: str = Field(default="", description="生成模型名称")
    final_answer: str = Field(default="", description="最终输出的回答文本")
    cited_chunk_ids: list[str] = Field(default_factory=list, description="回答中引用的[[id]]列表")
    context_chunk_count: int = Field(default=0, description="实际拼入Prompt的上下文切片数")
    ttft_ms: float = Field(default=0.0, description="首字延迟TTFT(毫秒)")
    timing: ModuleTiming = Field(default_factory=ModuleTiming, description="模块耗时与网络IO")


class RAGTraceRecord(BaseModel):
    """职责: 完整记录单次RAG查询全链路的所有观测数据"""

    trace_id: str = Field(..., description="唯一追踪ID(对应task_id)")
    session_id: str = Field(default="", description="会话ID")
    folder_hour: str = Field(..., description="归类文件夹名称, 格式 yy-mm-dd:hh")
    created_at: str = Field(..., description="创建时间(YYYY-MM-DD HH:MM:SS)")
    timestamp: float = Field(default=0.0, description="Unix时间戳(秒)")
    total_duration_ms: float = Field(default=0.0, description="端到端总耗时(毫秒)")
    status: str = Field(default="processing", description="状态: processing | completed | failed")
    error_message: str = Field(default="", description="错误信息(如有)")

    query_module: QueryModuleTrace = Field(default_factory=QueryModuleTrace, description="1. Query改写与意图提取模块")
    router_module: RouterModuleTrace = Field(default_factory=RouterModuleTrace, description="2. 置信度评估与路由模块")
    retrieval_module: RetrievalModuleTrace = Field(default_factory=RetrievalModuleTrace, description="3. Dense/Sparse多路检索模块")
    rrf_module: RrfModuleTrace = Field(default_factory=RrfModuleTrace, description="4. RRF倒排融合模块")
    reranker_module: RerankerModuleTrace = Field(default_factory=RerankerModuleTrace, description="5. Reranker重排模块")
    cliff_module: CliffModuleTrace = Field(default_factory=CliffModuleTrace, description="6. 断崖检测截断模块")
    output_module: OutputModuleTrace = Field(default_factory=OutputModuleTrace, description="7. 最终模型输出模块")


class TraceSummaryItem(BaseModel):
    """职责: 用于侧边栏文件夹树展示的单条Trace轻量摘要"""

    trace_id: str = Field(..., description="唯一追踪ID")
    session_id: str = Field(default="", description="会话ID")
    folder_hour: str = Field(..., description="所属文件夹 yy-mm-dd:hh")
    created_at: str = Field(..., description="创建时间字符串")
    timestamp: float = Field(default=0.0, description="时间戳")
    original_query: str = Field(default="", description="原始Query")
    rewritten_query: str = Field(default="", description="改写后Query")
    item_names: list[str] = Field(default_factory=list, description="对齐主体名称")
    confidence_score: float = Field(default=0.0, description="最高置信度分数")
    route_taken: str = Field(default="", description="实际走的路由分支")
    total_duration_ms: float = Field(default=0.0, description="端到端总耗时(毫秒)")
    status: str = Field(default="completed", description="执行状态")


class TraceFolderGroup(BaseModel):
    """职责: 表示单个 yy-mm-dd:hh 归档文件夹及其包含的Trace摘要列表"""

    folder_hour: str = Field(..., description="文件夹名称, 格式 yy-mm-dd:hh")
    count: int = Field(default=0, description="该文件夹下的Trace总数")
    traces: list[TraceSummaryItem] = Field(default_factory=list, description="按时间倒序排列的Trace摘要列表")
