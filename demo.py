import argparse
import asyncio
import json

from runtime import (
    AgentRuntime,
    Message,
    ModelReply,
    ToolCall,
    ToolInput,
    ToolRegistry,
    ToolSpec,
)


class AddInput(ToolInput):
    a: int
    b: int


async def add(arguments: AddInput) -> int:
    return arguments.a + arguments.b


class ScriptedModel:
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict],
    ) -> ModelReply:
        last = messages[-1]

        # 第一轮：看见用户消息后，请求调用工具。
        if last.role == "user":
            if not any(tool["name"] == "add" for tool in tools):
                raise ValueError("模型没有看到 add 工具")

            return ModelReply(
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="add",
                        arguments={"a": 7, "b": 5},
                    ),
                )
            )

        # 第二轮：必须读取实际工具结果后再回答。
        if last.role == "tool" and last.result is not None:
            request = messages[-2].tool_calls[0]

            if request.call_id != last.result.call_id:
                raise ValueError("工具结果与请求不匹配")

            if last.result.is_error:
                return ModelReply(
                    text=f"工具执行失败：{last.result.content}"
                )

            return ModelReply(text=str(last.result.content))

        raise ValueError("不符合预期的消息顺序")


def build_runtime(model=None, handler=add) -> AgentRuntime:
    tools = ToolRegistry()

    tools.register(
        ToolSpec(
            name="add",
            description="计算两个整数之和",
            input_type=AddInput,
            handler=handler,
        )
    )

    return AgentRuntime(
        model if model is not None else ScriptedModel(),
        tools,
    )


async def main():
    parser = argparse.ArgumentParser(description="第一课：最小 Agent Runtime")
    parser.add_argument("--offline", action="store_true", help="使用离线模拟模型")
    parser.add_argument("--prompt", default="请使用 add 工具计算 7 与 5 的和")
    args = parser.parse_args()
    if args.offline:
        result = await build_runtime().run(args.prompt)
    else:
        from deepseek_model import DeepSeekModel
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
