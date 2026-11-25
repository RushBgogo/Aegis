"""审批绑定实际参数；执行器负责超时、取消和持久化执行账本。"""
import asyncio
import hashlib
import json
import time

from artex.recovery import OperationLedger


class ApprovalDenied(RuntimeError):
    """审批拒绝。"""


class ApprovalExpired(RuntimeError):
    """审批过期。"""


def fingerprint(tool: str, args: dict) -> str:
    """工具名 + 参数决定一份审批；参数变化必须重新审批。"""
    raw = json.dumps(
        [tool, args],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class ApprovalService:
    def __init__(self, repo, clock=None):
        self.repo = repo
        self.clock = clock or time.time

    def request(self, task_id, operation_id, tool, args, ttl=60):
        if ttl <= 0:
            raise ValueError("审批有效期必须大于 0")

        digest = fingerprint(tool, args)
        value = {
            "tool": tool,
            "args": args,
            "digest": digest,
            "state": "pending",
            "expires_at": self.clock() + ttl,
            "operation_id": operation_id,
        }
        self.repo.put_record("approval", operation_id, value, task_id=task_id)
        existing = self.repo.get_record(
            "approval", operation_id, task_id=task_id
        )
        if existing is None:
            raise RuntimeError("审批记录写入失败")
        if existing["value"]["digest"] != digest:
            raise ValueError("operation_id 对应的参数发生变化，旧审批不可复用")
        return existing["value"]

    def decide(self, task_id, operation_id, allowed):
        record = self.repo.get_record(
            "approval", operation_id, task_id=task_id
        )
        if record is None:
            raise KeyError(operation_id)

        value = record["value"]
        if self.clock() >= value["expires_at"]:
            raise ApprovalExpired("审批已过期")
        if value["state"] != "pending":
            raise ValueError("审批已处理")

        value["state"] = "approved" if allowed else "denied"
        if not self.repo.put_record(
            "approval",
            operation_id,
            value,
            task_id=task_id,
            expected_version=record["version"],
        ):
            raise RuntimeError("审批被并发处理，请重新读取")

    async def wait(self, task_id, operation_id, digest):
        """等待审批决定；短轮询便于本课程观察，数据库仍是权威。"""
        while True:
            record = self.repo.get_record(
                "approval", operation_id, task_id=task_id
            )
            if record is None or record["value"]["digest"] != digest:
                raise ApprovalDenied("审批记录不存在或参数不一致")

            value = record["value"]
            if self.clock() >= value["expires_at"]:
                raise ApprovalExpired("审批已过期，未执行工具")
            if value["state"] == "approved":
                return
            if value["state"] == "denied":
                raise ApprovalDenied("操作被拒绝，未执行工具")
            await asyncio.sleep(0.02)


class ApprovalExecutor:
    """在真正执行工具之前统一做策略、审批、任务状态复核和账本保护。"""

    def __init__(self, repo, ledger=None, policy=None, timeout=30):
        self.repo = repo
        self.ledger = ledger or OperationLedger(repo)
        self.approvals = ApprovalService(repo)
        self.policy = policy or (
            lambda tool, args: "allow" if tool == "observe" else "ask"
        )
        self.timeout = timeout

    async def execute(self, task_id, operation_id, tool, args, handler):
        action = self.policy(tool, args)
        if action not in ("allow", "deny", "ask"):
            raise ValueError("策略只能返回 allow/deny/ask")

        self.repo.append_event(
            task_id,
            "tool_policy",
            dict(operation_id=operation_id, tool=tool, action=action),
        )
        # 策略日志仍然记录，但已经不在运行态的任务不能产生新副作用。
        if self.repo.get_task(task_id)["status"] != "running":
            raise ApprovalDenied("任务当前不在运行态")

        if action == "deny":
            raise ApprovalDenied("策略拒绝执行")

        if action == "ask":
            approval = self.approvals.request(task_id, operation_id, tool, args)
            await asyncio.wait_for(
                self.approvals.wait(task_id, operation_id, approval["digest"]),
                self.timeout,
            )

        # 等待审批期间任务可能被暂停、取消或删除，必须再次复核。
        if self.repo.get_task(task_id)["status"] != "running":
            raise ApprovalDenied("任务当前不在运行态")

        try:
            result = await asyncio.wait_for(
                self.ledger.execute(
                    task_id, operation_id, tool, args, handler
                ),
                self.timeout,
            )
        except (asyncio.CancelledError, TimeoutError):
            # 账本保留 started，不能把不确定执行宣称为成功。
            raise

        self.repo.append_event(
            task_id,
            "tool_committed",
            dict(operation_id=operation_id, tool=tool),
        )
        return result


async def run_json_process(argv, arguments, timeout=5):
    """固定 argv、无 shell；脚本从 stdin 接收 JSON，从 stdout 返回 JSON。"""
    if not argv:
        raise ValueError("argv 不能为空")

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(
                json.dumps(arguments, ensure_ascii=False).encode()
            ),
            timeout,
        )
        if process.returncode:
            detail = stderr.decode(errors="replace").strip()[:500]
            raise RuntimeError(
                f"工具进程失败，exit={process.returncode}: {detail}"
            )
        return json.loads(stdout.decode())
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
