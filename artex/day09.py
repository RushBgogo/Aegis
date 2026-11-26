"""Day 9 模型策略演示：共享预算、分类重试与熔断；默认离线。"""
import argparse
import asyncio
import json

from artex.db import Repository
from artex.model_policy import (
    Budget,
    DeepSeekUsageModel,
    GovernedModel,
    ProviderFailure,
)
from runtime import (
    AgentRuntime,
    Message,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)


class ObserveInput(ToolInput):
    target: str


async def observe(arguments):
    return {
        "target": arguments.target,
        "status": 200,
        "body": "public sample",
    }


class MeteredFixture:
    """离线模型：第一次模拟 429，第二次成功并返回固定 usage。"""

    def __init__(self):
        self.attempt = 0
        self.last_usage = 0

    async def complete(self, messages, tools):
        self.attempt += 1
        if self.attempt == 1:
            raise ProviderFailure(429, "offline rate limit")
        self.last_usage = 20

        last = messages[-1]
        if last.role == "tool" and last.result is not None:
            return ModelReply(
                text=json.dumps(last.result.content, ensure_ascii=False)
            )
        return ModelReply(
            tool_calls=(
                ToolCall(
                    "metered-1",
                    "observe",
                    {"target": "https://lab.example/public"},
                ),
            )
        )


def build_runtime(model):
    tools = ToolRegistry()
    tools.register(
        ToolSpec("observe", "读取固定样例目标", ObserveInput, observe)
    )
    return AgentRuntime(model, tools)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Aegis Day9 模型预算/重试/熔断演示"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用真实 DeepSeek（读取 config.local.json）",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="真实模型使用流式响应",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task("模型策略", "演示预算与重试")
        repo.transition_task(task["id"], "created", "running")
        budget = Budget(repo, task["id"], 20000)
        inner = (
            DeepSeekUsageModel(stream=args.stream)
            if args.live
            else MeteredFixture()
        )

        async with GovernedModel(inner, budget, reservation=2048) as model:
            result = await build_runtime(model).run(
                "请使用 observe 读取固定样例目标"
            )

        print("答案：", result.text)
        print("终态：", result.reason)
        state = budget.snapshot()
        print(
            "预算：spent=",
            state["spent"],
            "holds=",
            state["holds"],
            "uncertain=",
            len(state["uncertain"]),
        )
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
