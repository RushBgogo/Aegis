"""Day 10 上下文演示：原始事实保留，摘要只是可失效缓存。"""
import argparse
import asyncio
import json

from artex.context import ContextBuilder, ContextualModel, DigestStore
from artex.db import Repository
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
    return {
        "target": arguments.target,
        "status": 403,
        "body": "public sample is protected",
    }


class ScriptedObserveModel:
    """离线模型：先请求 observe，再根据工具结果给出摘要。"""

    async def complete(self, messages, tools):
        last = messages[-1]
        if last.role == "user":
            return ModelReply(
                tool_calls=(
                    ToolCall(
                        "context-1",
                        "observe",
                        {"target": "https://lab.example/public"},
                    ),
                )
            )
        if last.role == "tool" and last.result is not None:
            return ModelReply(
                text=json.dumps(last.result.content, ensure_ascii=False)
            )
        raise ValueError("unexpected message order")


def build_runtime(model):
    tools = ToolRegistry()
    tools.register(
        ToolSpec("observe", "读取固定样例目标", ObserveInput, observe)
    )
    return AgentRuntime(model, tools)


def build_parser():
    parser = argparse.ArgumentParser(description="Aegis Day10 上下文演示")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="上下文硬字符预算",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be > 0")

    repo = Repository("sqlite:///:memory:")
    repo.setup()  # 演示专用内存库；生产数据库使用 alembic upgrade head
    try:
        task = repo.create_task(
            "上下文演示",
            "核实固定样例目标",
            scope=["https://lab.example/public"],
        )
        task_id = task["id"]
        repo.transition_task(task_id, "created", "running")
        fact = repo.create_node(
            task_id,
            "fact",
            "样例返回 403：目标存在但未确认问题",
            payload={"status": 403},
        )

        async def summarize(nodes):
            return "已观察到 403；目标存在，但尚未确认可利用问题。"

        store = DigestStore(repo)
        digest = await store.generate(task_id, [fact["id"]], summarize)

        builder = ContextBuilder(repo)
        model = ContextualModel(
            ScriptedObserveModel(),
            builder,
            task_id,
            max_chars=args.max_chars,
        )
        result = await build_runtime(model).run("请观察固定样例目标")

        print("摘要：", digest["body"] if digest else "未生成")
        print("Runtime 结果：", result.text)
        print(
            "原始事实：",
            repo.get_node(task_id, fact["id"])["text"],
        )
    finally:
        repo.close()


if __name__ == "__main__":
    asyncio.run(main())
