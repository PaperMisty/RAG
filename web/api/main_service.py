import sys
from pathlib import Path

# 将工作区根目录加入 Python 寻址路径
sys.path.append(str(Path(__file__).parents[2]))

import uuid
import json
import time
import aiofiles
from datetime import datetime
from queue import Queue

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Body, Path
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from atguigu.tool.task_utils import (
    TASK_STATUS_FAILED,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_PROCESSING,
    update_task_status,
    get_task_info,
    add_done_task,
    add_running_task,
)
from atguigu.config.config import RAW_DIR, STATIC_DIR, OUTPUT_DIR, MinIOConfig
from atguigu.import_process.main_graph import GraphRunner
from atguigu.query_process.main_graph import QueryMainGraphRunner
from atguigu.tool.minio_client_tool import get_minio_client
from atguigu.tool.mongo_client_tool import clear_history, get_recent_history_list, add_or_update_data
from atguigu.tool.logger import logger

app = FastAPI(
    title="掌柜智库一体化服务",
    description="整合文档导入与智能问答检索的一体化后端服务",
    version="0.1.0",
)

# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- 1. 通用路由与心跳 -----------------


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ----------------- 2. 问答对话检索服务 -----------------


@app.get("/history/{session_id}")
async def get_history(session_id: str = Path(..., description="会话ID")):
    """
    获取历史记录
    """
    history_lst = get_recent_history_list(session_id, limit=100, reverse=False)  # 50 轮内刷新不丢旧轮；此前 limit=10 导致超 5 轮后旧消息在刷新后不可见
    for item in history_lst:
        item["_id"] = str(item["_id"])
    print("get", session_id)
    return {"items": history_lst}


@app.delete("/history/{session_id}")
async def delete_history(session_id: str = Path(..., description="会话ID")):
    """
    删除历史记录
    """
    clear_history(session_id)
    print("clear:", session_id)
    return {"status": "ok"}


class Query(BaseModel):
    query: str = Field(..., description="问题")
    session_id: str = Field(..., description="会话ID")


queue_dict = {}


def run_query_graph(task_id, original_query, session_id):
    if not queue_dict.get(task_id):
        queue_dict[task_id] = Queue()
    q = queue_dict[task_id]
    init_state = {"task_id": task_id, "original_query": original_query, "session_id": session_id, "q": q}
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        q.put({"event": "progress", "data": get_task_info(task_id)})
        QueryMainGraphRunner.create_and_run(init_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        q.put({"event": "progress", "data": get_task_info(task_id)})
    except Exception as e:
        import traceback
        import sys

        print(f"问答检索工作流执行报错: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        update_task_status(task_id, TASK_STATUS_FAILED)
        q.put({"event": "error", "data": get_task_info(task_id)})
        # 图在 AnswerOutput 写 assistant 之前失败时，Mongo 里只剩入口处写的 user 记录，
        # 刷新后就是“只剩用户消息”。补一条失败提示记录（非编造答案），使刷新后可见失败态。
        try:
            add_or_update_data({
                "session_id": session_id,
                "task_id": task_id,
                "role": "assistant",
                "text": "（本次查询失败，未生成答案，请重试）",
                "chunks": [],
                "ts": time.time(),
            })
        except Exception:
            pass


@app.post("/query")
async def query(background_tasks: BackgroundTasks, query: Query = Body(..., description="查询请求体参数")):
    original_query = query.query
    session_id = query.session_id
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_query_graph, task_id, original_query, session_id)
    return {"task_id": task_id, "original_query": original_query, "session_id": session_id}


def generate_stream(task_id):
    while not queue_dict.get(task_id):
        time.sleep(0.1)
    q = queue_dict.get(task_id)
    try:
        while True:
            data = q.get()
            yield f"event: {data.get('event')}\n"
            yield f"data: {json.dumps(data.get('data'), ensure_ascii=False)}\n\n"

            event_type = data.get("event")
            data_content = data.get("data")
            status = data_content.get("status") if isinstance(data_content, dict) else None

            if event_type == "error" or event_type == "final" or status in ["completed", "failed"]:
                break
    except GeneratorExit:
        pass
    except Exception as e:
        try:
            logger.error(f"流式推送中遇到未知异常: {e}")
        except:
            pass
    finally:
        if task_id in queue_dict:
            del queue_dict[task_id]


@app.get("/stream/{task_id}")
async def stream(task_id: str = Path(..., description="任务ID")):
    return StreamingResponse(generate_stream(task_id), media_type="text/event-stream")


# ----------------- 3. 知识库文档导入服务 -----------------


def run_import_graph(task_id: str, file_path: str, file_path_output: str):
    try:
        init_state = {"task_id": task_id, "local_file_path": file_path, "local_dir": file_path_output}
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        GraphRunner.create_and_run(init_state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        import traceback
        import sys

        print(f"主图导入工作流执行失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        update_task_status(task_id, TASK_STATUS_FAILED)
        raise Exception(f"主图执行失败,错误信息:{e}")


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(..., description="文件描述")):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(add_running_task, task_id, "upload_file")

    upload_dir = RAW_DIR / datetime.now().strftime("%Y%m%d")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            await f.write(chunk)
    logger.info(f"文件上传服务器成功:{file_path}")

    background_tasks.add_task(add_done_task, task_id, "upload_file")

    minio_client = get_minio_client()
    minio_client.fput_object(
        bucket_name=MinIOConfig.minio_bucket_name,
        object_name=f"pdf_file/{datetime.now().strftime('%Y%m%d')}/{task_id}/{file.filename}",
        file_path=str(file_path),
    )

    logger.info(
        f"文件上传到MinIO成功,路径: {MinIOConfig.minio_bucket_name + '/' + 'pdf_file' + '/' + datetime.now().strftime('%Y%m%d') + '/' + task_id + '/' + file.filename}"
    )

    dir_name = OUTPUT_DIR / datetime.now().strftime("%Y%m%d")
    dir_name.mkdir(parents=True, exist_ok=True)

    file_path_output = OUTPUT_DIR / dir_name
    background_tasks.add_task(run_import_graph, task_id, str(file_path), str(file_path_output))

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    return get_task_info(task_id)


# ----------------- 4. 挂载静态目录与运行入口 -----------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_service:app", host="127.0.0.1", port=8001, reload=True)
