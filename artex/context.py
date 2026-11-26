"""原始数据是事实，摘要只是带版本的可失效缓存。"""
import json
from dataclasses import asdict
from uuid import uuid4

from runtime import Message


class ContextOverflow(ValueError):
    """不可丢内容自身超过预算，需要缩小任务或扩大窗口。"""


class DigestStore:
    def __init__(self, repo):
        self.repo = repo

    def fresh(self, task_id, value):
        """源节点版本必须完全匹配；活跃意图不能被视为冷历史。"""
        for node_id, version in value["members"].items():
            try:
                node = self.repo.get_node(task_id, node_id)
            except KeyError:
                return False
            if node["version"] != version:
                return False
            if (
                node["kind"] == "intent"
                and node["state"] in ("open", "running")
            ):
                return False
        return True

    async def generate(self, task_id, member_ids, summarizer):
        """生成摘要，但绝不覆盖原始事实。"""
        members = [self.repo.get_node(task_id, item) for item in member_ids]
        if not members:
            raise ValueError("摘要至少需要一个源节点")
        if any(
            node["kind"] in ("goal", "hint")
            or (
                node["kind"] == "intent"
                and node["state"] in ("open", "running")
            )
            for node in members
        ):
            raise ValueError("目标、约束提示和活跃意图不能进入冷摘要")

        snapshot = {node["id"]: node["version"] for node in members}
        body = await summarizer(members)
        if not isinstance(body, str) or not body.strip():
            raise ValueError("摘要正文不能为空")
        value = {
            "members": snapshot,
            "body": body,
            "id": uuid4().hex,
        }
        # 写入前再次核对，源事实永远不被摘要覆盖。
        if not self.fresh(task_id, value):
            return None
        if not self.repo.put_record(
            "digest", value["id"], value, task_id=task_id
        ):
            raise RuntimeError("摘要 ID 冲突")
        return value

    def valid(self, task_id):
        return [
            row["value"]
            for row in self.repo.list_records("digest", task_id=task_id)
            if self.fresh(task_id, row["value"])
        ]


class ContextBuilder:
    """把任务上下文分成 pinned、digest、recent 三层。"""

    def __init__(self, repo):
        self.repo = repo
        self.digests = DigestStore(repo)

    def build(self, task_id, max_chars=6000):
        task = self.repo.get_task(task_id)
        nodes = self.repo.list_nodes(task_id)
        active = [
            node
            for node in nodes
            if node["kind"] == "intent"
            and node["state"] in ("open", "running")
        ]
        pinned = {
            "task_id": task_id,
            "goal": task["goal"],
            "scope": task["scope"],
            "constraints": [
                node["text"]
                for node in nodes
                if node["kind"] == "hint"
                and node["payload"].get("constraint")
            ],
            "active_intents": [
                {"id": node["id"], "text": node["text"]}
                for node in active
            ],
            "goals": [
                {
                    "id": node["id"],
                    "text": node["text"],
                    "state": node["state"],
                }
                for node in nodes
                if node["kind"] == "goal"
            ],
        }
        result = {"pinned": pinned, "digests": [], "recent": [], "omitted": 0}

        def size():
            return len(
                json.dumps(result, ensure_ascii=False, sort_keys=True)
            )

        if size() > max_chars:
            raise ContextOverflow(
                "目标/范围/约束/活跃意图已超预算；拒绝静默截断"
            )

        covered = set()
        for digest in self.digests.valid(task_id):
            result["digests"].append(digest)
            if size() > max_chars:
                result["digests"].pop()
            else:
                covered.update(digest["members"])

        for node in reversed(nodes):
            if node["kind"] not in ("fact", "finding", "intent"):
                continue
            if node["id"] in covered:
                continue
            if (
                node["kind"] == "intent"
                and node["state"] in ("open", "running")
            ):
                continue
            row = {
                "id": node["id"],
                "kind": node["kind"],
                "text": node["text"],
                "version": node["version"],
                "state": node["state"],
            }
            result["recent"].append(row)
            if size() > max_chars:
                result["recent"].pop()
                result["omitted"] += 1

        # omitted 的位数变化也需要计入硬字符预算。
        while size() > max_chars and result["recent"]:
            result["recent"].pop()
            result["omitted"] += 1
        if size() > max_chars:
            raise ContextOverflow("上下文元数据超预算")
        return result

    def render(self, task_id, max_chars=6000):
        return json.dumps(
            self.build(task_id, max_chars),
            ensure_ascii=False,
            sort_keys=True,
        )


class ContextualModel:
    """给底层模型注入结构化任务记忆，但不改写 Runtime 的消息历史。"""

    def __init__(self, inner, builder, task_id, max_chars=12000):
        self.inner = inner
        self.builder = builder
        self.task_id = task_id
        self.max_chars = max_chars

    async def __aenter__(self):
        enter = getattr(self.inner, "__aenter__", None)
        if enter:
            await enter()
        return self

    async def __aexit__(self, *args):
        exit_method = getattr(self.inner, "__aexit__", None)
        if exit_method:
            await exit_method(*args)

    @property
    def last_usage(self):
        """让外层 GovernedModel 看到底层真实 usage。"""
        return getattr(self.inner, "last_usage", None)

    async def complete(self, messages, tools):
        overhead = len(
            json.dumps(
                [asdict(message) for message in messages],
                ensure_ascii=False,
            )
        )
        overhead += len(json.dumps(tools, ensure_ascii=False))
        context = self.builder.render(
            self.task_id, self.max_chars - overhead
        )
        return await self.inner.complete(
            [
                Message(
                    role="system",
                    content="结构化任务记忆：" + context,
                ),
                *messages,
            ],
            tools,
        )


def model_summarizer(model):
    async def summarize(nodes):
        prompt = (
            "仅总结已发生的结果，保留否定与不确定结论，不添加新事实。源节点："
            + json.dumps(
                [
                    {
                        "id": node["id"],
                        "text": node["text"],
                        "payload": node["payload"],
                    }
                    for node in nodes
                ],
                ensure_ascii=False,
            )
        )
        reply = await model.complete(
            [Message(role="user", content=prompt)], []
        )
        if reply.tool_calls or not reply.text:
            raise ValueError("摘要模型应返回纯文本")
        return reply.text

    return summarize
