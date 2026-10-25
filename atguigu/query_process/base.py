# atguigu/query_process/base.py

"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from atguigu.tool.task_utils import add_done_task
from atguigu.tool.task_utils import get_task_info
from atguigu.tool.task_utils import add_running_task
from atguigu.tool.task_utils import add_node_duration
from abc import abstractmethod, ABC
import time

from atguigu.query_process.state import QueryGraphState
from atguigu.tool.logger import logger


class NodeBase(ABC):

    name: str = "node_base"

    def __init__(self):
        """
        强制子类设置name
        """
        if self.name == "node_base":
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")

    def __call__(self, state: QueryGraphState):
        """
        节点执行入口
        """
        # Task/session are logged on every node line: without them,
        # concurrent queries interleave in one log file and no line
        # can be attributed to the query the user is staring at.
        task_id = state.get("task_id")
        session_id = state.get("session_id")
        ctx = f"task={task_id} session={session_id}"
        started = time.time()
        try:
            logger.info(f"[{self.name}] 开始执行... {ctx}")
            q = state.get("q")
            add_running_task(task_id, self.name)
            if q is not None:
                q.put({"event": "progress", "data": get_task_info(task_id)})

            result = self.process(state)

            elapsed = time.time() - started
            add_node_duration(task_id, self.name, elapsed)
            add_done_task(task_id, self.name)
            if q is not None:
                q.put({"event": "progress", "data": get_task_info(task_id)})

            logger.info(f"[{self.name}] 结束执行... {ctx} elapsed={elapsed:.1f}s")

            return result
        except Exception as e:
            logger.error(
                f"[{self.name}] 执行失败: {ctx} elapsed={time.time() - started:.1f}s "
                f"error_type={type(e).__name__} error={str(e)[:500]}"
            )
            raise

    @abstractmethod
    def process(self, state: QueryGraphState):
        """
        节点的核心处理逻辑
        :return:
        """
        pass
