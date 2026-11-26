"""20 个固定行为场景。用例编号和 fixture 版本是评估结果的一部分。"""
import time

from artex.approval import ApprovalDenied, ApprovalExecutor
from artex.context import DigestStore
from artex.db import Repository
from artex.engine import Engine
from artex.model_policy import Budget, BudgetExceeded
from artex.planner import Planner
from artex.worker import Worker
from runtime import (
    AgentRuntime,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)


class ObserveInput(ToolInput):
    target: str


async def observe(arguments):
    return {"target": arguments.target, "status": 403}


def build_runtime(model):
    """固定 observe 工具；评估不依赖 demo.py 的旧示例。"""
    tools = ToolRegistry()
    tools.register(
        ToolSpec("observe", "读取固定样例目标", ObserveInput, observe)
    )
    return AgentRuntime(model, tools)


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)

    async def complete(self, messages, tools):
        item = next(self.replies)
        if isinstance(item, Exception):
            raise item
        return item


async def evaluate():
    results = []
    started = time.perf_counter()

    # 八个 Runtime 协议场景，错误也是有预期结果的用例。
    runtime_cases = [
        ("answer", [ModelReply(text="ok")], "completed", None),
        ("empty", [ModelReply()], "protocol_error", None),
        (
            "both",
            [
                ModelReply(
                    text="bad",
                    tool_calls=(
                        ToolCall(
                            "x", "observe", {"target": "public"}
                        ),
                    ),
                )
            ],
            "protocol_error",
            None,
        ),
        (
            "model_error",
            [RuntimeError("offline model")],
            "model_error",
            None,
        ),
        (
            "unknown_tool",
            [
                ModelReply(
                    tool_calls=(ToolCall("x", "missing", {}),)
                ),
                ModelReply(text="error seen"),
            ],
            "completed",
            "unknown_tool",
        ),
        (
            "missing_arg",
            [
                ModelReply(
                    tool_calls=(ToolCall("x", "observe", {}),)
                ),
                ModelReply(text="error seen"),
            ],
            "completed",
            "invalid_arguments",
        ),
        (
            "strict_type",
            [
                ModelReply(
                    tool_calls=(
                        ToolCall("x", "observe", {"target": 1}),
                    )
                ),
                ModelReply(text="error seen"),
            ],
            "completed",
            "invalid_arguments",
        ),
        (
            "duplicate_id",
            [
                ModelReply(
                    tool_calls=(
                        ToolCall(
                            "x", "observe", {"target": "public"}
                        ),
                    )
                )
            ]
            * 2,
            "protocol_error",
            None,
        ),
    ]
    for name, replies, expected, code in runtime_cases:
        outcome = await build_runtime(SequenceModel(replies)).run(name)
        observed_codes = [
            message.result.content.get("code")
            for message in outcome.messages
            if message.result and message.result.is_error
        ]
        results.append(
            {
                "case": name,
                "passed": outcome.reason == expected
                and (code is None or code in observed_codes),
                "reason": outcome.reason,
                "model_turns": outcome.turns,
            }
        )

    # 八个端到端样例，结果必须绑定真实 evidence。
    for index in range(8):
        repo = Repository("sqlite:///:memory:")
        repo.setup()
        try:
            target = f"https://lab.example/eval/{index}"
            issue = index % 3 == 0
            fixtures = {
                target: {
                    "status": 200 if issue else 403,
                    "body": f"fixture-{index}",
                    "issue": issue,
                }
            }
            task = repo.create_task(
                "offline evaluation", "核实样例行为", [target]
            )["id"]
            await Engine(
                repo, worker=Worker(repo, fixtures=fixtures)
            ).run(task)

            facts = repo.list_nodes(task, kind="fact")
            findings = repo.list_nodes(task, kind="finding")
            valid = 0
            for fact in facts:
                row = repo.get_record(
                    "evidence",
                    fact["payload"]["evidence_id"],
                    task_id=task,
                )
                valid += bool(
                    row
                    and row["value"]["target"] == target
                    and row["value"]["body"]
                    == fixtures[target]["body"]
                    and row["value"]["call_id"]
                    == fact["payload"]["call_id"]
                )
            passed = (
                repo.get_task(task)["status"] == "done"
                and len(facts) == 1
                and valid == 1
                and len(findings) == int(issue)
            )
            results.append(
                {
                    "case": f"e2e_{index}",
                    "passed": passed,
                    "evidence_checked": len(facts),
                    "evidence_valid": valid,
                    "expected_findings": int(issue),
                    "findings": len(findings),
                }
            )
        finally:
            repo.close()

    repo = Repository("sqlite:///:memory:")
    repo.setup()
    try:
        task = repo.create_task(
            "service evaluation",
            "核实样例",
            ["https://lab.example/public"],
        )["id"]
        repo.transition_task(task, "created", "running")
        planner = Planner(repo)
        await planner.plan(task)
        await planner.plan(task)
        results.append(
            {
                "case": "planner_dedup",
                "passed": len(repo.list_nodes(task, kind="intent")) == 1,
            }
        )

        called = []

        async def handler():
            called.append(True)
            return {}

        denied = False
        try:
            await ApprovalExecutor(
                repo, policy=lambda tool, args: "deny"
            ).execute(task, "eval-op", "write", {}, handler)
        except ApprovalDenied:
            denied = True
        results.append(
            {
                "case": "approval_deny",
                "passed": denied and not called,
            }
        )

        budget = Budget(repo, task, 100)
        budget.reserve("first", 80)
        blocked = False
        try:
            budget.reserve("second", 80)
        except BudgetExceeded:
            blocked = True
        results.append(
            {
                "case": "budget_no_double_spend",
                "passed": blocked,
            }
        )

        fact = repo.create_node(task, "fact", "old")

        async def stale(nodes):
            repo.update_node(
                task, fact["id"], "new", {}, fact["version"]
            )
            return "old summary"

        digest = await DigestStore(repo).generate(
            task, [fact["id"]], stale
        )
        results.append(
            {
                "case": "digest_stale_rejected",
                "passed": digest is None,
            }
        )
    finally:
        repo.close()

    checked = sum(
        item.get("evidence_checked", 0) for item in results
    )
    valid = sum(
        item.get("evidence_valid", 0) for item in results
    )
    return {
        "fixture_version": "study-v1",
        "model": "deterministic-fixtures",
        "cases": results,
        "count": len(results),
        "passed": sum(item["passed"] for item in results),
        "evidence_valid_rate": (
            valid / checked if checked else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "live_api_calls": 0,
    }
