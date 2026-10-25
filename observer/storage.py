# observer/storage.py

import json
from pathlib import Path
import aiofiles
from atguigu.tool.logger import logger
from observer.schemas import RAGTraceRecord, TraceFolderGroup, TraceSummaryItem

# 观测数据归档根目录: observer/traces/
TRACES_ROOT_DIR: Path = Path(__file__).parent / "traces"
TRACES_ROOT_DIR.mkdir(parents=True, exist_ok=True)


def get_folder_path(folder_hour: str) -> Path:
    """职责: 获取并创建指定 yy-mm-dd:hh 归档文件夹路径"""
    # 过滤非法路径分隔符防止目录穿越
    safe_folder = folder_hour.replace("/", "_").replace("\\", "_").replace("..", "_")
    folder_path = TRACES_ROOT_DIR / safe_folder
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def save_trace_sync(trace: RAGTraceRecord) -> Path:
    """职责: 在后台工作线程中将Trace记录同步原子落盘到 yy-mm-dd:hh 目录"""
    # 1. 确定目标文件夹与临时文件路径
    folder_path = get_folder_path(trace.folder_hour)
    file_path = folder_path / f"{trace.trace_id}.json"
    tmp_path = folder_path / f".{trace.trace_id}.json.tmp"

    # 2. 序列化为格式化JSON并原子替换写入
    payload = trace.model_dump(mode="json")
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_path.write_text(json_text, encoding="utf-8")
    tmp_path.replace(file_path)
    return file_path


async def save_trace_async(trace: RAGTraceRecord) -> Path:
    """职责: 异步协程方式将Trace记录落盘到 yy-mm-dd:hh 目录"""
    # 1. 确定目标文件夹路径
    folder_path = get_folder_path(trace.folder_hour)
    file_path = folder_path / f"{trace.trace_id}.json"

    # 2. 使用 aiofiles 异步写入文件
    payload = trace.model_dump(mode="json")
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json_text)
    return file_path


def _build_summary_from_dict(data: dict) -> TraceSummaryItem:
    """职责: 从Trace原始字典提取轻量级摘要对象"""
    query_mod = data.get("query_module") or {}
    router_mod = data.get("router_module") or {}
    return TraceSummaryItem(
        trace_id=str(data.get("trace_id", "")),
        session_id=str(data.get("session_id", "")),
        folder_hour=str(data.get("folder_hour", "")),
        created_at=str(data.get("created_at", "")),
        timestamp=float(data.get("timestamp", 0.0) or 0.0),
        original_query=str(query_mod.get("original_query", "")),
        rewritten_query=str(query_mod.get("rewritten_query", "")),
        item_names=list(query_mod.get("aligned_item_names") or query_mod.get("extracted_item_names") or []),
        confidence_score=float(router_mod.get("confidence_score", 0.0) or 0.0),
        route_taken=str(router_mod.get("route_taken", "")),
        total_duration_ms=float(data.get("total_duration_ms", 0.0) or 0.0),
        status=str(data.get("status", "completed")),
    )


async def list_trace_folders_async() -> list[TraceFolderGroup]:
    """职责: 异步扫描所有 yy-mm-dd:hh 文件夹并返回按时间倒序排列的Trace分组列表"""
    if not TRACES_ROOT_DIR.exists():
        return []

    # 1. 获取所有子文件夹并按名称倒序排列(即最新小时在前)
    dirs = [d for d in TRACES_ROOT_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    dirs.sort(key=lambda p: p.name, reverse=True)

    folder_groups: list[TraceFolderGroup] = []
    # 2. 遍历每个小时文件夹读取其中的Trace JSON文件
    for folder_dir in dirs:
        json_files = [f for f in folder_dir.glob("*.json") if not f.name.startswith(".")]
        summaries: list[TraceSummaryItem] = []
        for jf in json_files:
            try:
                async with aiofiles.open(jf, "r", encoding="utf-8") as f:
                    raw_text = await f.read()
                data = json.loads(raw_text)
                summaries.append(_build_summary_from_dict(data))
            except Exception as e:
                logger.warning(f"读取Trace文件失败 path={jf}: {e}")

        # 3. 文件夹内按时间戳倒序排列
        summaries.sort(key=lambda item: item.timestamp, reverse=True)
        if summaries:
            folder_groups.append(
                TraceFolderGroup(
                    folder_hour=folder_dir.name,
                    count=len(summaries),
                    traces=summaries,
                )
            )

    return folder_groups


async def load_trace_detail_async(trace_id: str, folder_hour: str | None = None) -> dict | None:
    """职责: 根据 trace_id 异步加载单条完整 Trace JSON 数据"""
    safe_id = trace_id.replace("/", "").replace("\\", "").replace("..", "")
    # 1. 若指定了归属文件夹则直接定位读取
    if folder_hour:
        candidate = get_folder_path(folder_hour) / f"{safe_id}.json"
        if candidate.exists():
            async with aiofiles.open(candidate, "r", encoding="utf-8") as f:
                return json.loads(await f.read())

    # 2. 未指定文件夹时遍历各 yy-mm-dd:hh 目录查找
    if not TRACES_ROOT_DIR.exists():
        return None
    for folder_dir in sorted(TRACES_ROOT_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder_dir.is_dir():
            continue
        candidate = folder_dir / f"{safe_id}.json"
        if candidate.exists():
            async with aiofiles.open(candidate, "r", encoding="utf-8") as f:
                return json.loads(await f.read())
    return None


async def delete_trace_async(trace_id: str) -> bool:
    """职责: 异步删除指定 trace_id 对应的 JSON 文件及空文件夹"""
    safe_id = trace_id.replace("/", "").replace("\\", "").replace("..", "")
    if not TRACES_ROOT_DIR.exists():
        return False

    # 遍历定位并移除文件
    for folder_dir in TRACES_ROOT_DIR.iterdir():
        if not folder_dir.is_dir():
            continue
        candidate = folder_dir / f"{safe_id}.json"
        if candidate.exists():
            candidate.unlink(missing_ok=True)
            # 若文件夹已空则顺带清理空目录
            remaining = [f for f in folder_dir.glob("*.json") if not f.name.startswith(".")]
            if not remaining:
                try:
                    folder_dir.rmdir()
                except OSError:
                    pass
            return True
    return False
