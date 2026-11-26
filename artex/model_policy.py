"""供应商协议、错误策略、预算分层：使用量记账不等于美元计费。"""
import asyncio
import json
import time
from uuid import uuid4

from deepseek_model import DeepSeekModel
from runtime import ModelReply, ToolCall


class BudgetExceeded(RuntimeError):
    """任务预算不足。"""


class ProviderFailure(RuntimeError):
    def __init__(self, status, message="provider failure"):
        super().__init__(message)
        self.status_code = status


class Budget:
    """任务级 token 预算：reserve → settle / release / uncertain。

    holds 表示正在调用、尚未结算的预留；spent 是已确认的实际用量；
    uncertain 表示调用已发出但无法确认服务端消耗的预留。
    """

    def __init__(self, repo, task_id, limit=20000):
        if limit < 1:
            raise ValueError("预算应为正整数")
        self.repo, self.task_id = repo, task_id
        repo.put_record(
            "budget",
            "tokens",
            {"limit": limit, "spent": 0, "holds": {}, "uncertain": []},
            task_id=task_id,
        )

    def snapshot(self):
        return self.repo.get_record(
            "budget", "tokens", task_id=self.task_id
        )["value"]

    def _change(self, change):
        """用 CAS 循环修改预算；change 只负责修改传入的 dict。"""
        for _ in range(30):
            record = self.repo.get_record(
                "budget", "tokens", task_id=self.task_id
            )
            value = record["value"]
            change(value)
            if self.repo.put_record(
                "budget",
                "tokens",
                value,
                task_id=self.task_id,
                expected_version=record["version"],
            ):
                return value
        raise RuntimeError("预算并发冲突过多")

    def reserve(self, request_id, upper_bound):
        if upper_bound < 1:
            raise ValueError("预留额度必须大于 0")

        def change(value):
            if request_id in value["holds"]:
                raise ValueError("请求已预留")
            available = value["limit"] - value["spent"] - sum(
                value["holds"].values()
            )
            if available < upper_bound:
                raise BudgetExceeded("任务剩余额度不足，未调用模型")
            value["holds"][request_id] = upper_bound

        self._change(change)

    def settle(self, request_id, actual):
        if actual < 0:
            raise ValueError("实际用量不能小于 0")

        def change(value):
            if request_id not in value["holds"]:
                raise ValueError("找不到预留，禁止重复结算")
            del value["holds"][request_id]
            value["spent"] += actual
            value["uncertain"] = [
                item for item in value["uncertain"] if item != request_id
            ]

        self._change(change)

    def release(self, request_id):
        self._change(lambda value: value["holds"].pop(request_id, None))

    def mark_uncertain(self, request_id):
        def change(value):
            if (
                request_id in value["holds"]
                and request_id not in value["uncertain"]
            ):
                value["uncertain"].append(request_id)

        self._change(change)


def status_of(error):
    """把异常归类成可重试的 HTTP 状态，未知错误返回 None。"""
    status = getattr(error, "status_code", None)
    if status is not None:
        return status
    if isinstance(error, (TimeoutError, ConnectionError)):
        return 503

    # openai SDK 的超时/连接异常继承自 APIConnectionError。
    from openai import APIConnectionError

    return 503 if isinstance(error, APIConnectionError) else None


class Circuit:
    """按任务持久化的小型熔断器，重启后仍保留冷却状态。"""

    def __init__(self, repo, task_id, threshold=3, cooldown=10):
        self.repo, self.task_id = repo, task_id
        self.threshold, self.cooldown = threshold, cooldown

    def state(self):
        row = self.repo.get_record("circuit", "model", task_id=self.task_id)
        return row or {"value": {"failures": 0, "until": 0}, "version": 0}

    def check(self):
        if self.state()["value"]["until"] > time.time():
            raise RuntimeError("模型暂时熔断，冷却后重试")

    def result(self, success):
        for _ in range(30):
            row = self.state()
            failures = 0 if success else row["value"]["failures"] + 1
            value = {
                "failures": failures,
                "until": (
                    time.time() + self.cooldown
                    if failures >= self.threshold
                    else 0
                ),
            }
            if self.repo.put_record(
                "circuit",
                "model",
                value,
                task_id=self.task_id,
                expected_version=row["version"],
            ):
                return
        raise RuntimeError("熔断状态并发冲突")


class GovernedModel:
    """在真实模型外层统一处理预算、分类重试与熔断。"""

    def __init__(
        self,
        inner,
        budget,
        max_attempts=3,
        reservation=2048,
        delay=0.05,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        self.inner = inner
        self.budget = budget
        self.max_attempts = max_attempts
        self.reservation = reservation
        self.delay = delay
        self.circuit = Circuit(budget.repo, budget.task_id)

    async def __aenter__(self):
        enter = getattr(self.inner, "__aenter__", None)
        if enter:
            await enter()
        return self

    async def __aexit__(self, *args):
        exit_method = getattr(self.inner, "__aexit__", None)
        if exit_method:
            await exit_method(*args)

    async def complete(self, messages, tools):
        for attempt in range(self.max_attempts):
            self.circuit.check()
            request_id = uuid4().hex
            self.budget.reserve(request_id, self.reservation)
            try:
                reply = await self.inner.complete(messages, tools)
            except asyncio.CancelledError:
                # 调用已发出，无法确认服务端是否消耗，保留 uncertain 预留。
                self.budget.mark_uncertain(request_id)
                raise
            except Exception as error:
                status = status_of(error)
                if status in (400, 401, 402, 403, 404, 422, 429):
                    self.budget.release(request_id)
                else:
                    # 网络断开/5xx 无法证明服务端没有生成，保留预留。
                    self.budget.mark_uncertain(request_id)

                retry = status == 429 or (
                    status is not None and status >= 500
                )
                self.circuit.result(False)
                self.budget.repo.append_event(
                    self.budget.task_id,
                    "model_failure",
                    {
                        "status": status,
                        "retry": retry,
                        "attempt": attempt + 1,
                    },
                )
                if not retry or attempt + 1 == self.max_attempts:
                    raise
                await asyncio.sleep(self.delay * (2 ** attempt))
            else:
                usage = getattr(self.inner, "last_usage", None)
                if usage is None:
                    self.budget.mark_uncertain(request_id)
                else:
                    self.budget.settle(request_id, usage)
                self.circuit.result(True)
                return reply


class DeepSeekUsageModel(DeepSeekModel):
    """复用第一课配置/客户端；新增真实 usage 和可选流式拼接。"""

    DEFAULT_SYSTEM_PROMPT = "按任务要求使用可用工具，并根据真实工具结果回答。"

    def __init__(self, stream=False, system_prompt=None):
        super().__init__(system_prompt=system_prompt)
        self.stream = stream
        self.last_usage = None

    def messages(self, messages):
        prompt = (
            getattr(self, "system_prompt", None)
            or self.DEFAULT_SYSTEM_PROMPT
        )
        result = [{"role": "system", "content": prompt}]
        for message in messages:
            if message.tool_calls:
                result.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(
                                        call.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            elif message.role == "tool":
                tool_result = message.result
                if tool_result is None:
                    raise ValueError("tool 消息缺少结果")
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.call_id,
                        "content": json.dumps(
                            {
                                "ok": not tool_result.is_error,
                                "content": tool_result.content,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            else:
                result.append(
                    {"role": message.role, "content": message.content}
                )
        return result

    async def complete(self, messages, tools):
        self.last_usage = None
        request = dict(
            model=self.model,
            messages=self.messages(messages),
            stream=self.stream,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if tools:
            request["tools"] = [
                {"type": "function", "function": tool} for tool in tools
            ]
        if self.stream:
            request["stream_options"] = {"include_usage": True}

        response = await self.client.chat.completions.create(**request)
        if not self.stream:
            self.last_usage = (
                response.usage.total_tokens if response.usage else None
            )
            if not response.choices:
                raise ValueError("空模型响应")
            choice = response.choices[0]
            if choice.finish_reason not in ("stop", "tool_calls"):
                raise ValueError("模型响应被截断或未完成")
            message = choice.message
            if message.tool_calls:
                return ModelReply(
                    tool_calls=tuple(
                        ToolCall(
                            call.id,
                            call.function.name,
                            self.parse(call.function.arguments),
                        )
                        for call in message.tool_calls
                    )
                )
            return ModelReply(text=message.content)

        text, calls, terminal = [], {}, None
        try:
            async for chunk in response:
                if chunk.usage:
                    self.last_usage = chunk.usage.total_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                terminal = choice.finish_reason or terminal
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                for call in delta.tool_calls or []:
                    item = calls.setdefault(
                        call.index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    item["id"] += call.id or ""
                    if call.function:
                        item["name"] += call.function.name or ""
                        item["arguments"] += call.function.arguments or ""
        finally:
            await response.close()

        if terminal not in ("stop", "tool_calls"):
            raise ValueError("流式响应未正常结束，不执行不完整工具参数")
        if calls:
            return ModelReply(
                tool_calls=tuple(
                    ToolCall(
                        item["id"],
                        item["name"],
                        self.parse(item["arguments"]),
                    )
                    for _, item in sorted(calls.items())
                )
            )
        return ModelReply(text="".join(text))

    @staticmethod
    def parse(raw):
        arguments = json.loads(raw)
        if not isinstance(arguments, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        return arguments
