# observer/collector.py

from contextlib import contextmanager
from datetime import datetime
import re
import threading
import time
from typing import Any, Generator

from atguigu.tool.logger import logger
from observer.schemas import (
    CandidateItemScore,
    IOBreakdownItem,
    ModuleTiming,
    RAGTraceRecord,
    RouteBranchStatus,
    TraceChunkItem,
)
from observer.storage import save_trace_sync


class TraceCollector:
    """职责: 线程安全地收集单次RAG查询全链路各模块的观测数据与网络IO耗时"""

    def __init__(self) -> None:
        """职责: 初始化内存态Trace字典与线程互斥锁"""
        self._traces: dict[str, RAGTraceRecord] = {}
        self._lock = threading.RLock()

    def start_trace(self, task_id: str, original_query: str, session_id: str) -> RAGTraceRecord:
        """职责: 在查询入口处创建并初始化一条新的Trace记录"""
        if not task_id:
            task_id = f"local_{int(time.time() * 1000)}"
        now = datetime.now()
        folder_hour = now.strftime("%y-%m-%d:%H")
        created_at = now.strftime("%Y-%m-%d %H:%M:%S")
        ts = time.time()

        # 构造初始三路路由线状态(默认未激活)
        default_routes = [
            RouteBranchStatus(
                route_key="high_hybrid",
                label="高置信度路由 (score ≥ 0.8)",
                condition="confidence ≥ 0.8 或唯一中置信候选",
                is_active=False,
                target_desc="Dense + Sparse + HyDE + Web 多路检索 → RRF → Reranker → 断崖检测 → LLM",
            ),
            RouteBranchStatus(
                route_key="medium_clarify",
                label="中置信度路由 (0.6 < score < 0.8, 多候选)",
                condition="0.6 < confidence < 0.8 且候选主体 > 1",
                is_active=False,
                target_desc="跳过向量切片检索，直接输出候选澄清反问选项",
            ),
            RouteBranchStatus(
                route_key="low_web_only",
                label="低置信度路由 (score ≤ 0.6 / 无匹配)",
                condition="confidence ≤ 0.6 或知识库无匹配主体",
                is_active=False,
                target_desc="跳过本地知识库检索，仅走 Tavily 网络搜索兜底总结",
            ),
        ]

        record = RAGTraceRecord(
            trace_id=task_id,
            session_id=session_id or "",
            folder_hour=folder_hour,
            created_at=created_at,
            timestamp=ts,
            status="processing",
        )
        record.query_module.original_query = original_query
        record.router_module.routes = default_routes

        with self._lock:
            self._traces[task_id] = record
            self._flush_safe(record)
        return record

    def get_trace(self, task_id: str | None) -> RAGTraceRecord | None:
        """职责: 获取内存中正在追踪的Trace对象"""
        if not task_id:
            return None
        with self._lock:
            return self._traces.get(task_id)

    def _get_target_timing(self, record: RAGTraceRecord, module_key: str) -> ModuleTiming:
        """职责: 根据模块标识定位对应的ModuleTiming对象"""
        mapping = {
            "query": record.query_module.timing,
            "router": record.router_module.timing,
            "retrieval": record.retrieval_module.timing,
            "dense": record.retrieval_module.dense_timing,
            "sparse": record.retrieval_module.sparse_timing,
            "hyde": record.retrieval_module.hyde_timing,
            "web": record.retrieval_module.web_timing,
            "rrf": record.rrf_module.timing,
            "reranker": record.reranker_module.timing,
            "cliff": record.cliff_module.timing,
            "output": record.output_module.timing,
        }
        return mapping.get(module_key, record.query_module.timing)

    @contextmanager
    def track_module(self, task_id: str | None, module_key: str) -> Generator[None, None, None]:
        """职责: 上下文管理器，测量整个模块方格的执行总耗时"""
        t0 = time.perf_counter()
        started_ts = time.time()
        try:
            yield
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            ended_ts = time.time()
            record = self.get_trace(task_id)
            if record:
                with self._lock:
                    timing = self._get_target_timing(record, module_key)
                    timing.started_at = started_ts
                    timing.ended_at = ended_ts
                    timing.total_ms = round(timing.total_ms + elapsed_ms, 2)

    @contextmanager
    def track_io(
        self,
        task_id: str | None,
        module_key: str,
        name: str,
        io_type: str = "network",
        detail: str = "",
    ) -> Generator[None, None, None]:
        """职责: 上下文管理器，测量单次网络IO或数据库调用耗时并写入对应模块"""
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception as exc:
            status = "timeout" if "timeout" in str(exc).lower() or "超时" in str(exc) else "error"
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            record = self.get_trace(task_id)
            if record:
                with self._lock:
                    timing = self._get_target_timing(record, module_key)
                    timing.io_breakdown.append(
                        IOBreakdownItem(
                            name=name,
                            io_type=io_type,
                            duration_ms=elapsed_ms,
                            status=status,
                            detail=detail,
                        )
                    )

    def add_io_item(
        self,
        task_id: str | None,
        module_key: str,
        name: str,
        duration_ms: float,
        io_type: str = "network",
        status: str = "ok",
        detail: str = "",
    ) -> None:
        """职责: 手动向指定模块追加一条IO耗时记录"""
        record = self.get_trace(task_id)
        if not record:
            return
        with self._lock:
            timing = self._get_target_timing(record, module_key)
            timing.io_breakdown.append(
                IOBreakdownItem(
                    name=name,
                    io_type=io_type,
                    duration_ms=round(duration_ms, 2),
                    status=status,
                    detail=detail,
                )
            )

    def set_module_total_ms(self, task_id: str | None, module_key: str, total_ms: float) -> None:
        """职责: 直接设置指定模块的总耗时(毫秒)"""
        record = self.get_trace(task_id)
        if not record:
            return
        with self._lock:
            timing = self._get_target_timing(record, module_key)
            timing.total_ms = round(total_ms, 2)

    def record_query_and_router(
        self,
        task_id: str | None,
        original_query: str,
        rewritten_query: str,
        key_words: list[str],
        extracted_item_names: list[str],
        aligned_item_names: list[str],
        matched_results: list[dict],
        answer: str,
        is_low_confidence: bool,
        history_turns: int = 0,
    ) -> None:
        """职责: 记录Query改写模块与置信度路由模块的完整数据"""
        record = self.get_trace(task_id)
        if not record:
            return

        # 1. 整理候选主体列表与最高置信度得分
        candidates: list[CandidateItemScore] = []
        seen_names: set[str] = set()
        max_score = 0.0

        for item_info in matched_results or []:
            item_name_val = (item_info.get("entity") or {}).get("item_name", "")
            if not item_name_val or item_name_val.strip() in ["", "空字符串", "None", "null"]:
                continue
            dist = float(item_info.get("distance", 0.0) or 0.0)
            if dist > max_score:
                max_score = dist
            if item_name_val not in seen_names:
                seen_names.add(item_name_val)
                level = "high" if dist >= 0.8 else ("medium" if dist > 0.6 else "low")
                candidates.append(
                    CandidateItemScore(
                        item_name=item_name_val,
                        score=round(dist, 4),
                        level=level,
                    )
                )

        candidates.sort(key=lambda x: x.score, reverse=True)

        # 2. 判定实际走的路由线与另外两条灰化路由线
        if answer:
            route_taken = "medium_clarify"
            route_reason = f"存在 {len(candidates)} 个中置信度候选主体 (0.6 < score < 0.8)，触发用户澄清确认"
        elif is_low_confidence:
            route_taken = "low_web_only"
            route_reason = f"最高主体匹配分 {max_score:.3f} ≤ 0.6（或无匹配），路由至纯网络搜索分支"
        else:
            route_taken = "high_hybrid"
            route_reason = f"最高主体匹配分 {max_score:.3f} 达到高置信度确认条件，激活混合检索全链路"

        with self._lock:
            # 填充 Query 模块
            record.query_module.original_query = original_query
            record.query_module.rewritten_query = rewritten_query
            record.query_module.key_words = key_words
            record.query_module.extracted_item_names = extracted_item_names
            record.query_module.aligned_item_names = aligned_item_names
            record.query_module.history_turns = history_turns

            # 填充 Router 模块
            record.router_module.confidence_score = round(max_score, 4)
            record.router_module.route_taken = route_taken
            record.router_module.route_reason = route_reason
            record.router_module.matched_candidates = candidates
            for r in record.router_module.routes:
                r.is_active = r.route_key == route_taken

            self._flush_safe(record)

    def record_dense_sparse_retrieval(
        self,
        task_id: str | None,
        dense_chunks: list[dict],
        sparse_chunks: list[dict],
        hybrid_chunks: list[dict],
    ) -> None:
        """职责: 记录Dense稠密检索、Sparse稀疏检索及混合检索各自的排名切片"""
        record = self.get_trace(task_id)
        if not record:
            return

        def _to_trace_chunks(raw_list: list[dict], src_label: str) -> list[TraceChunkItem]:
            items: list[TraceChunkItem] = []
            for idx, ch in enumerate(raw_list or [], 1):
                items.append(
                    TraceChunkItem(
                        rank=idx,
                        chunk_id=str(ch.get("id", "")),
                        file_title=str(ch.get("file_title", "")),
                        section_title=str(ch.get("section_title") or ch.get("title") or ""),
                        item_name=str(ch.get("item_name", "")),
                        content=str(ch.get("chunk_content") or ch.get("content") or ""),
                        source=src_label,
                        score=round(float(ch.get("score", 0.0) or 0.0), 4),
                    )
                )
            return items

        with self._lock:
            record.retrieval_module.executed = True
            record.retrieval_module.dense_chunks = _to_trace_chunks(dense_chunks, "dense")
            record.retrieval_module.sparse_chunks = _to_trace_chunks(sparse_chunks, "sparse")
            record.retrieval_module.hybrid_chunks = _to_trace_chunks(hybrid_chunks, "local")
            self._flush_safe(record)

    def record_hyde_retrieval(
        self,
        task_id: str | None,
        hyde_answer: str,
        hyde_chunks: list[dict],
    ) -> None:
        """职责: 记录HyDE假设性文档生成内容与检索排名切片"""
        record = self.get_trace(task_id)
        if not record:
            return

        items: list[TraceChunkItem] = []
        for idx, ch in enumerate(hyde_chunks or [], 1):
            items.append(
                TraceChunkItem(
                    rank=idx,
                    chunk_id=str(ch.get("id", "")),
                    file_title=str(ch.get("file_title", "")),
                    section_title=str(ch.get("section_title") or ch.get("title") or ""),
                    item_name=str(ch.get("item_name", "")),
                    content=str(ch.get("chunk_content") or ch.get("content") or ""),
                    source="hyde",
                    score=round(float(ch.get("score", 0.0) or 0.0), 4),
                )
            )

        with self._lock:
            record.retrieval_module.executed = True
            record.retrieval_module.hyde_query = hyde_answer or ""
            record.retrieval_module.hyde_chunks = items
            self._flush_safe(record)

    def record_web_retrieval(self, task_id: str | None, web_docs: list[dict]) -> None:
        """职责: 记录Tavily网络搜索返回的文档列表"""
        record = self.get_trace(task_id)
        if not record:
            return

        items: list[TraceChunkItem] = []
        for idx, doc in enumerate(web_docs or [], 1):
            items.append(
                TraceChunkItem(
                    rank=idx,
                    chunk_id=f"web_{idx}",
                    file_title=str(doc.get("title", "网络搜索结果")),
                    section_title=str(doc.get("url", "")),
                    item_name="Web",
                    content=str(doc.get("content", "")),
                    url=str(doc.get("url", "")),
                    source="web",
                    score=round(1.0 / idx, 4),
                )
            )

        with self._lock:
            record.retrieval_module.executed = True
            record.retrieval_module.web_docs = items
            self._flush_safe(record)

    def record_rrf(
        self,
        task_id: str | None,
        rrf_chunks: list[dict],
        top_k: int = 10,
        rrf_smooth: int = 60,
    ) -> None:
        """职责: 记录RRF倒排融合结果，并将排在TopK之后的切片标记为灰色"""
        record = self.get_trace(task_id)
        if not record:
            return

        with self._lock:
            # 建立 Dense / Sparse / HyDE 排名反查表以便在RRF卡片上展示来源排名
            dense_rank_map = {c.chunk_id: c.rank for c in record.retrieval_module.dense_chunks if c.chunk_id}
            sparse_rank_map = {c.chunk_id: c.rank for c in record.retrieval_module.sparse_chunks if c.chunk_id}
            hyde_rank_map = {c.chunk_id: c.rank for c in record.retrieval_module.hyde_chunks if c.chunk_id}

            items: list[TraceChunkItem] = []
            for idx, ch in enumerate(rrf_chunks or [], 1):
                cid = str(ch.get("id", ""))
                is_gray = idx > top_k
                gray_reason = f"排名 #{idx} 超出 RRF Top-{top_k} 阈值，被挤出候选核心区" if is_gray else ""
                items.append(
                    TraceChunkItem(
                        rank=idx,
                        chunk_id=cid,
                        file_title=str(ch.get("file_title", "")),
                        section_title=str(ch.get("section_title") or ch.get("title") or ""),
                        item_name=str(ch.get("item_name", "")),
                        content=str(ch.get("chunk_content") or ch.get("content") or ""),
                        source=str(ch.get("source", "local")),
                        score=round(float(ch.get("score_rrf", 0.0) or 0.0), 5),
                        raw_score=round(float(ch.get("score", 0.0) or 0.0), 4),
                        dense_rank=dense_rank_map.get(cid),
                        sparse_rank=sparse_rank_map.get(cid),
                        hyde_rank=hyde_rank_map.get(cid),
                        rrf_rank=idx,
                        is_grayed=is_gray,
                        gray_reason=gray_reason,
                    )
                )

            total_cnt = len(items)
            kept_cnt = min(total_cnt, top_k)
            gray_cnt = max(0, total_cnt - kept_cnt)

            record.rrf_module.executed = True
            record.rrf_module.rrf_smooth = rrf_smooth
            record.rrf_module.top_k = top_k
            record.rrf_module.total_unique_count = total_cnt
            record.rrf_module.kept_count = kept_cnt
            record.rrf_module.grayed_count = gray_cnt
            record.rrf_module.chunks = items
            self._flush_safe(record)

    def record_rerank_and_cliff(
        self,
        task_id: str | None,
        rerank_all_docs: list[dict],
        cliff_kept_docs: list[dict],
        abs_gap_thres: float = 0.25,
        ratio_gap_thres: float = 0.10,
        min_len: int = 3,
        max_len: int = 10,
    ) -> None:
        """职责: 记录Reranker重排完整列表及断崖检测的截断位置与灰化切片"""
        record = self.get_trace(task_id)
        if not record:
            return

        with self._lock:
            kept_len = len(cliff_kept_docs or [])
            total_len = len(rerank_all_docs or [])

            rerank_items: list[TraceChunkItem] = []
            cliff_items: list[TraceChunkItem] = []

            cliff_triggered = False
            trigger_reason = ""
            cliff_after_rank = kept_len

            for idx, doc in enumerate(rerank_all_docs or [], 1):
                cur_score = float(doc.get("score", 0.0) or 0.0)
                cid = str(doc.get("id", "") or f"doc_{idx}")
                file_title = str(doc.get("file_title") or doc.get("title") or "")
                section_title = str(doc.get("section_title") or doc.get("title") or "")
                item_name = str(doc.get("item_name") or doc.get("title") or "")
                content = str(doc.get("content") or doc.get("chunk_content") or "")
                url = str(doc.get("url") or "")
                source = str(doc.get("source") or "local")
                rrf_rank = doc.get("rrf_rank")

                # 1. Reranker 阶段项(全部激活展示)
                rerank_items.append(
                    TraceChunkItem(
                        rank=idx,
                        chunk_id=cid,
                        file_title=file_title,
                        section_title=section_title,
                        item_name=item_name,
                        content=content,
                        url=url,
                        source=source,
                        score=round(cur_score, 4),
                        rrf_rank=rrf_rank,
                        is_grayed=False,
                    )
                )

                # 2. 计算与前一名切片的绝对分差和相对分差
                abs_gap_val = None
                ratio_gap_val = None
                if idx > 1:
                    prev_score = float(rerank_all_docs[idx - 2].get("score", 0.0) or 0.0)
                    abs_gap_val = round(prev_score - cur_score, 4)
                    ratio_gap_val = round(abs_gap_val / prev_score, 4) if prev_score > 1e-9 else 0.0

                is_dropped = idx > kept_len
                gray_reason = ""
                if is_dropped:
                    if idx == kept_len + 1 and kept_len < min(max_len, total_len):
                        cliff_triggered = True
                        reasons = []
                        if abs_gap_val is not None and abs_gap_val > abs_gap_thres:
                            reasons.append(f"绝对分差 Δabs={abs_gap_val:.3f} > {abs_gap_thres}")
                        if ratio_gap_val is not None and ratio_gap_val > ratio_gap_thres:
                            reasons.append(f"相对分差 Δratio={ratio_gap_val * 100:.1f}% > {ratio_gap_thres * 100:.0f}%")
                        trigger_reason = "；".join(reasons) or f"在第 #{kept_len} 名后触发断崖截断"
                        gray_reason = f"触发断崖截断（{trigger_reason}）"
                    elif kept_len >= max_len and idx > max_len:
                        gray_reason = f"超出最大保留长度 MAX_LEN={max_len}"
                    else:
                        gray_reason = f"位于断崖截断点 (#{kept_len}) 之后，已被淘汰"

                cliff_items.append(
                    TraceChunkItem(
                        rank=idx,
                        chunk_id=cid,
                        file_title=file_title,
                        section_title=section_title,
                        item_name=item_name,
                        content=content,
                        url=url,
                        source=source,
                        score=round(cur_score, 4),
                        rrf_rank=rrf_rank,
                        is_grayed=is_dropped,
                        gray_reason=gray_reason,
                        abs_gap=abs_gap_val,
                        ratio_gap=ratio_gap_val,
                    )
                )

            if not cliff_triggered and total_len > max_len:
                trigger_reason = f"未触发分差断崖，按上限 MAX_LEN={max_len} 截断"
            elif not cliff_triggered:
                trigger_reason = f"分差平稳，保留全部 {kept_len} 条候选切片"

            record.reranker_module.executed = True
            record.reranker_module.input_count = total_len
            record.reranker_module.chunks = rerank_items

            record.cliff_module.executed = True
            record.cliff_module.abs_gap_threshold = abs_gap_thres
            record.cliff_module.ratio_gap_threshold = ratio_gap_thres
            record.cliff_module.min_len = min_len
            record.cliff_module.max_len = max_len
            record.cliff_module.cliff_triggered = cliff_triggered
            record.cliff_module.cliff_after_rank = cliff_after_rank
            record.cliff_module.trigger_reason = trigger_reason
            record.cliff_module.kept_count = kept_len
            record.cliff_module.dropped_count = max(0, total_len - kept_len)
            record.cliff_module.chunks = cliff_items

            self._flush_safe(record)

    def record_output(
        self,
        task_id: str | None,
        final_answer: str,
        output_mode: str = "rag_llm",
        model_name: str = "",
        context_chunk_count: int = 0,
        ttft_ms: float = 0.0,
    ) -> None:
        """职责: 记录最终大模型输出文本、引用角标及首字延迟"""
        record = self.get_trace(task_id)
        if not record:
            return

        # 提取答案中的引用编号 [[1]], [[2]]
        cited_ids: list[str] = []
        for m in re.findall(r"\[\[(\d+)\]\]", final_answer or ""):
            if m not in cited_ids:
                cited_ids.append(m)

        with self._lock:
            record.output_module.executed = True
            record.output_module.output_mode = output_mode
            record.output_module.model_name = model_name
            record.output_module.final_answer = final_answer or ""
            record.output_module.cited_chunk_ids = cited_ids
            record.output_module.context_chunk_count = context_chunk_count
            record.output_module.ttft_ms = round(ttft_ms, 2)
            self._flush_safe(record)

    def finish_trace(
        self,
        task_id: str | None,
        status: str = "completed",
        error_message: str = "",
        total_duration_ms: float | None = None,
    ) -> RAGTraceRecord | None:
        """职责: 结束本次Trace追踪并将完整观测记录落盘保存"""
        record = self.get_trace(task_id)
        if not record:
            return None

        with self._lock:
            record.status = status
            if error_message:
                record.error_message = error_message
            if total_duration_ms is not None:
                record.total_duration_ms = round(total_duration_ms, 2)
            else:
                record.total_duration_ms = round((time.time() - record.timestamp) * 1000, 2)

            # 汇总检索模块总耗时(取各并发子模块最大耗时或累加)
            ret = record.retrieval_module
            if ret.executed and ret.timing.total_ms <= 0:
                sub_totals = [
                    ret.dense_timing.total_ms,
                    ret.sparse_timing.total_ms,
                    ret.hyde_timing.total_ms,
                    ret.web_timing.total_ms,
                ]
                ret.timing.total_ms = round(max(sub_totals) if any(sub_totals) else 0.0, 2)

            self._flush_safe(record)
            return record

    def _flush_safe(self, record: RAGTraceRecord) -> None:
        """职责: 安全地将当前Trace状态同步刷新到磁盘文件"""
        try:
            save_trace_sync(record)
        except Exception as e:
            logger.warning(f"保存Trace文件失败 trace_id={record.trace_id}: {e}")


# 全局单例收集器
trace_collector = TraceCollector()
