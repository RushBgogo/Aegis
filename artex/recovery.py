"""已完成操作可重放；外部副作用不确定时不能盲目重试。"""
import asyncio
from dataclasses import asdict
import hashlib
import json
from typing import Any, Awaitable, Callable

from runtime import Message, ToolCall, ToolResult


class UncertainOperation(RuntimeError):
    """外部写入结果未知，需要查询或人工确认。"""


class OperationLedger:
    """按 task_id + operation_id 记录副作用。

    - operation_id 是业务副作用 ID；call_id 只是模型工具协议 ID，不要混用；
    - started：handler 已开始，可能已经产生外部副作用；
    - done：handler 已完成，结果可以直接重放；
    - idempotent 以首次登记为准，后续调用不能把它改成 true。
    """

    def __init__(self, repo):
        self.repo = repo
        # 只协调同一个 OperationLedger 实例；跨实例保护依赖数据库 CAS。
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def execute(
        self,
        task_id: str,
        operation_id: str,
        tool: str,
        args: dict,
        handler: Callable[[], Awaitable[Any]],
        *,
        idempotent: bool = True,
    ) -> Any:
        """执行一次带账本保护的副作用。"""
        fingerprint = hashlib.sha256(
            json.dumps([tool, args], sort_keys=True).encode()
        ).hexdigest()
        lock = self._locks.setdefault((task_id, operation_id), asyncio.Lock())

        async with lock:
            record = self.repo.get_record("operation", operation_id, task_id)
            if record:
                value = record["value"]
                if value["fingerprint"] != fingerprint:
                    raise ValueError("operation_id reused with different arguments")
                if value["state"] == "done":
                    return value["result"]
                # 首次登记是否为幂等是权威值，不能让本次调用覆盖。
                if not value.get("idempotent", False):
                    raise UncertainOperation("operation started but outcome unknown")
                version = record["version"]
            else:
                value = {
                    "state": "started",
                    "tool": tool,
                    "args": args,
                    "fingerprint": fingerprint,
                    "idempotent": idempotent,
                }
                if not self.repo.put_record(
                    "operation", operation_id, value, task_id
                ):
                    raise RuntimeError(
                        "operation claimed concurrently; retry through owner"
                    )
                version = 1

            # 幂等工具必须自行按稳定 operation_id 去重；
            # 账本只能保证“已完成结果可重放”，不能消除执行后未提交的时间窗。
            result = await handler()
            done = {**value, "state": "done", "result": result}
            if not self.repo.put_record(
                "operation", operation_id, done, task_id, version
            ):
                raise RuntimeError("operation completion version conflict")
            return result


def save_checkpoint(
    repo,
    task_id: str,
    session_id: str,
    messages: list[Message],
) -> None:
    """保存可恢复的会话历史；工具请求和结果必须成对。"""
    if not messages:
        raise ValueError("checkpoint cannot be empty")

    requested, resolved = set(), set()
    for message in messages:
        requested.update(call.call_id for call in message.tool_calls)
        if message.result is not None:
            resolved.add(message.result.call_id)
    if requested != resolved:
        raise ValueError(
            "checkpoint must be taken after all tool results are recorded"
        )

    old = repo.get_record("checkpoint", session_id, task_id)
    payload = {"messages": [asdict(message) for message in messages]}
    if not repo.put_record(
        "checkpoint",
        session_id,
        payload,
        task_id,
        old["version"] if old else 0,
    ):
        raise RuntimeError("checkpoint version conflict")


def load_checkpoint(repo, task_id: str, session_id: str) -> list[Message] | None:
    """读取会话历史；没有检查点时返回 None。"""
    record = repo.get_record("checkpoint", session_id, task_id)
    if not record:
        return None

    messages = []
    for value in record["value"]["messages"]:
        calls = tuple(ToolCall(**call) for call in value["tool_calls"])
        result = ToolResult(**value["result"]) if value["result"] else None
        messages.append(Message(value["role"], value["content"], calls, result))
    return messages
