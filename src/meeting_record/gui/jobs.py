"""單任務狀態機與事件串流。

單人本機情境：同時只允許一個任務。事件累積在帶序號的 list，
SSE 客戶端從任意序號續讀，斷線重連不掉事件。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    IDLE = "idle"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    SUMMARIZING = "summarizing"
    DONE = "done"
    ERROR = "error"


ACTIVE_STATES = {
    JobState.EXTRACTING,
    JobState.TRANSCRIBING,
    JobState.DIARIZING,
    JobState.SUMMARIZING,
}


class JobBusyError(Exception):
    """已有任務執行中。"""


class JobManager:
    """管理單一背景任務：啟動、事件發布、狀態查詢。

    runner 由呼叫端注入（實際 pipeline 或測試 mock），
    在背景 thread 中以本 manager 為參數執行。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._state: JobState = JobState.IDLE
        self._events: list[dict[str, Any]] = []
        self._info: dict[str, Any] = {}
        self._thread: threading.Thread | None = None

    # -- 查詢 --

    @property
    def state(self) -> JobState:
        with self._cond:
            return self._state

    def snapshot(self) -> dict[str, Any]:
        """目前任務完整狀態（供頁面刷新後恢復畫面）。"""
        with self._cond:
            return {
                "state": self._state.value,
                "info": dict(self._info),
                "events": list(self._events),
            }

    def wait_events(self, after: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        """回傳序號 > after 的事件；沒有新事件時最多等 timeout 秒。"""
        with self._cond:
            if len(self._events) <= after + 1:
                self._cond.wait(timeout)
            return self._events[after + 1 :]

    # -- 事件發布（runner 在背景 thread 呼叫） --

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        with self._cond:
            self._events.append(
                {"seq": len(self._events), "type": event_type, "data": data or {}}
            )
            self._cond.notify_all()

    def set_state(self, state: JobState) -> None:
        with self._cond:
            self._state = state
            self._events.append(
                {"seq": len(self._events), "type": "state", "data": {"state": state.value}}
            )
            self._cond.notify_all()

    # -- 啟動 --

    def start(self, info: dict[str, Any], runner: Callable[[JobManager], None]) -> None:
        """啟動背景任務；已有任務執行中則丟 JobBusyError。"""
        with self._cond:
            if self._state in ACTIVE_STATES:
                raise JobBusyError("已有任務執行中，請等它完成後再開始新任務")
            self._state = JobState.IDLE
            self._events = []
            self._info = dict(info)
            self._thread = threading.Thread(
                target=self._run, args=(runner,), daemon=True, name="meeting-record-job"
            )
            self._thread.start()

    def _run(self, runner: Callable[[JobManager], None]) -> None:
        try:
            runner(self)
        except Exception as exc:  # runner 應自行處理已知錯誤；這是最後防線
            self.emit("error", {"message": str(exc)})
            self.set_state(JobState.ERROR)
