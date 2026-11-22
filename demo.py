import argparse
import asyncio
import json

from deepseek_model import DeepSeekModel

from runtime import (
    AgentRuntime,
    ModelClient,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)


class AddInput(ToolInput):
    a: int
    b: int


async def add(arguments: AddInput) -> int:
    return arguments.a + arguments.b


def build_runtime(model: ModelClient) -> AgentRuntime:
    tools = ToolRegistry()

    tools.register(
        ToolSpec(
            name="add",
            description="计算两个整数之和",
            input_type=AddInput,
            handler=add,
        )
    )

    return AgentRuntime(model, tools)


async def main():
    parser = argparse.ArgumentParser(description="Aegis：最小 Agent Runtime")
    parser.add_argument("--prompt", default="请使用 add 工具计算 7 与 5 的和")
    args = parser.parse_args()

    try:
        model = DeepSeekModel()
    except ValueError as exc:
        parser.error(str(exc))

    async with model:
        result = await build_runtime(model).run(args.prompt)

    for event in result.events:
        payload = json.dumps(event.data, ensure_ascii=False)

        print(
            event.seq,
            event.kind,
            event.call_id or "-",
            payload,
        )

    print(
        f"终态：{result.reason}，"
        f"答案：{result.text}，"
        f"模型轮数：{result.turns}"
    )
    if result.reason != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
