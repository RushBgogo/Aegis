"""查询、正式干预和旁路问答有各自的状态边界。"""
import asyncio
import json
import time
from collections import Counter
from contextlib import AsyncExitStack
from uuid import uuid4

from sqlalchemy import select, update

from artex.db import nodes, records, tasks
from runtime import Message


class HumanService:
    def __init__(self, repo):
        self.repo = repo
        self.locks = {}

    def summary(self, task_id):
        task = self.repo.get_task(task_id)
        all_nodes = self.repo.list_nodes(task_id)
        return {
            "task_id": task_id,
            "status": task["status"],
            "goal": task["goal"],
            "scope": task["scope"],
            "counts": dict(
                Counter(node["kind"] for node in all_nodes)
            ),
            "active": [
                {
                    "id": node["id"],
                    "text": node["text"],
                    "state": node["state"],
                }
                for node in all_nodes
                if node["kind"] == "intent"
                and node["state"] in ("open", "running")
            ],
            "facts": [
                {"id": node["id"], "text": node["text"]}
                for node in all_nodes
                if node["kind"] == "fact"
            ][-20:],
        }

    def add_hint(self, task_id, text, request_id):
        """同一 request_id 幂等；相同 ID 不同正文必须冲突。"""
        if not text.strip() or not request_id.strip():
            raise ValueError("提示和请求标识不能为空")

        body = {"kind": "hint", "text": text}
        old = self.repo.get_record(
            "human_request", request_id, task_id
        )
        if old and old["value"] != body:
            raise ValueError("相同请求标识不能提交不同内容")
        if not old and not self.repo.put_record(
            "human_request", request_id, body, task_id
        ):
            old = self.repo.get_record(
                "human_request", request_id, task_id
            )
            if old is None:
                raise RuntimeError("提示请求写入失败")
            if old["value"] != body:
                raise ValueError("相同请求标识不能提交不同内容")
        return self.repo.create_node(
            task_id, "hint", text, key="human-hint:" + request_id
        )

    async def inject_intent(
        self, task_id, target, text, request_id
    ):
        """任务行锁 + 请求账本 + 节点在同一事务中提交。"""
        # 单进程序列化相同任务人工入口；跨实例所有权留给 day19。
        lock = self.locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            with self.repo.write() as conn:
                task = conn.execute(
                    select(tasks)
                    .where(tasks.c.id == task_id)
                    .with_for_update()
                ).mappings().first()
                if task is None:
                    raise KeyError("task not found")
                if task["status"] in (
                    "deleting",
                    "deleted",
                    "cancelled",
                    "failed",
                ):
                    raise ValueError("任务当前状态不允许注入意图")
                if (
                    target not in task["scope"]
                    or not text.strip()
                    or not request_id.strip()
                ):
                    raise ValueError(
                        "人工意图需要范围内目标、文字和请求标识"
                    )

                body = {
                    "kind": "intent",
                    "text": text,
                    "target": target,
                }
                old = conn.execute(
                    select(records).where(
                        records.c.task_id == task_id,
                        records.c.namespace == "human_request",
                        records.c.key == request_id,
                    )
                ).mappings().first()
                if old and old["value"] != body:
                    raise ValueError(
                        "相同请求标识不能提交不同内容"
                    )

                node = conn.execute(
                    select(nodes).where(
                        nodes.c.task_id == task_id,
                        nodes.c.dedup_key
                        == "human-intent:" + request_id,
                    )
                ).mappings().first()
                if node:
                    return dict(node)

                value = {
                    "id": uuid4().hex,
                    "task_id": task_id,
                    "kind": "intent",
                    "text": text,
                    "dedup_key": "human-intent:" + request_id,
                    "state": "open",
                    "priority": 9,
                    "payload": {
                        "target": target,
                        "source": "human",
                    },
                    "version": 1,
                    "created_at": time.time(),
                }
                conn.execute(nodes.insert().values(**value))
                if not old:
                    conn.execute(
                        records.insert().values(
                            task_id=task_id,
                            namespace="human_request",
                            key=request_id,
                            value=body,
                            version=1,
                        )
                    )
                if task["status"] == "done":
                    conn.execute(
                        update(tasks)
                        .where(tasks.c.id == task_id)
                        .values(
                            status="running",
                            version=tasks.c.version + 1,
                        )
                    )
                return value

    def steer(self, task_id, intent_id, text, request_id):
        intent = self.repo.get_node(task_id, intent_id)
        if intent["kind"] != "intent" or intent["state"] not in (
            "open",
            "running",
        ):
            raise ValueError("只能纠偏待执行或正在执行的意图")
        if not text.strip():
            raise ValueError("纠偏文字不能为空")

        body = {"intent_id": intent_id, "text": text}
        old = self.repo.get_record("steering", request_id, task_id)
        if old:
            if (
                old["value"]["intent_id"] != intent_id
                or old["value"]["text"] != text
            ):
                raise ValueError(
                    "请求标识已被其他纠偏内容使用"
                )
            return old["value"]

        body["created_at"] = time.time()
        if not self.repo.put_record(
            "steering", request_id, body, task_id
        ):
            raise ValueError("纠偏正在提交，请查询后重试")
        return body

    async def ask(self, task_id, question, model=None):
        """使用快照构造独立对话，不写入 Worker 主消息或图。"""
        if not question.strip():
            raise ValueError("问题不能为空")
        snapshot = self.summary(task_id)
        if model is None:
            return {
                "answer": (
                    f"状态：{snapshot['status']}；"
                    f"节点统计：{snapshot['counts']}"
                ),
                "snapshot": snapshot,
            }

        prompt = json.dumps(
            {
                "question": question,
                "snapshot": snapshot,
                "instruction": (
                    "仅依据快照回答；不要调用工具或执行操作。"
                ),
            },
            ensure_ascii=False,
        )
        async with AsyncExitStack() as stack:
            if hasattr(model, "__aenter__"):
                model = await stack.enter_async_context(model)
            reply = await model.complete(
                [Message(role="user", content=prompt)], []
            )
        if reply.tool_calls or not reply.text:
            raise ValueError(
                "旁路问答不允许工具调用，且必须返回文本"
            )
        return {"answer": reply.text, "snapshot": snapshot}


class InterventionModel:
    """在下一次模型请求前插入纠偏，保留工具请求/结果顺序。"""

    def __init__(self, repo, inner):
        self.repo, self.inner = repo, inner
        self.delivered = set()

    async def __aenter__(self):
        if hasattr(self.inner, "__aenter__"):
            await self.inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if hasattr(self.inner, "__aexit__"):
            await self.inner.__aexit__(
                exc_type, exc_value, traceback
            )

    async def complete(self, messages, tools):
        first_user = next(
            (message for message in messages if message.role == "user"),
            None,
        )
        context = (
            json.loads(first_user.content) if first_user else {}
        )
        task_id = context.get("task_id")
        intent_id = context.get("intent_id")
        notes = []

        if task_id and intent_id:
            for record in self.repo.list_records(
                "steering", task_id
            ):
                value = record["value"]
                if (
                    value["intent_id"] == intent_id
                    and record["key"] not in self.delivered
                ):
                    notes.append(value["text"])
                    self.delivered.add(record["key"])

        if notes:
            messages = [
                Message(
                    role="system",
                    content="操作者纠偏：\n" + "\n".join(notes),
                ),
                *messages,
            ]
        return await self.inner.complete(messages, tools)
