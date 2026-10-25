# observer/router.py

from fastapi import APIRouter, HTTPException, Path, Query
from observer.collector import trace_collector
from observer.storage import (
    delete_trace_async,
    list_trace_folders_async,
    load_trace_detail_async,
)

router = APIRouter(prefix="/observer", tags=["RAG 链路观测层"])


@router.get("/folders")
async def get_trace_folders() -> dict:
    """职责: 获取按 yy-mm-dd:hh 归类的所有Trace文件夹及摘要列表"""
    folders = await list_trace_folders_async()
    return {
        "folders": [f.model_dump(mode="json") for f in folders],
        "total_folders": len(folders),
        "total_traces": sum(f.count for f in folders),
    }


@router.get("/trace/{trace_id}")
async def get_trace_detail(
    trace_id: str = Path(..., description="追踪ID(task_id)"),
    folder: str | None = Query(default=None, description="可选的归属文件夹 yy-mm-dd:hh"),
) -> dict:
    """职责: 获取单条Trace的完整全链路观测详情"""
    # 1. 优先检查内存中正在运行的实时Trace
    live_record = trace_collector.get_trace(trace_id)
    if live_record is not None:
        return live_record.model_dump(mode="json")

    # 2. 从磁盘归档目录异步加载
    detail = await load_trace_detail_async(trace_id=trace_id, folder_hour=folder)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"未找到 Trace 记录: {trace_id}")
    return detail


@router.delete("/trace/{trace_id}")
async def remove_trace(
    trace_id: str = Path(..., description="追踪ID(task_id)"),
) -> dict:
    """职责: 删除指定的单条Trace归档记录"""
    deleted = await delete_trace_async(trace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"未找到可删除的 Trace 记录: {trace_id}")
    return {"status": "ok", "trace_id": trace_id}
