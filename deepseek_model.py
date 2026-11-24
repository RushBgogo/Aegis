"""DeepSeek 的非思考、非流式工具调用适配器。"""

import json
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from runtime import Message, ModelReply, ToolCall


DEFAULT_SYSTEM_PROMPT = (
    "你是安全评估助手。只根据实际工具结果回答，不猜测证据；"
    "需要执行动作时调用提供的工具。"
)


class DeepSeekModel:
    def __init__(self, system_prompt=None):
        # 以代码文件定位配置，在 PyCharm 或其他工作目录启动也能读取。
        config_path = Path(__file__).with_name("config.local.json")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            raise ValueError("请将 config.example.json 复制为 config.local.json 并填写密钥") from None
        except (ValueError, UnicodeError):
            raise ValueError("config.local.json 不是有效的 UTF-8 JSON 配置") from None

        if not isinstance(config, dict):
            raise ValueError("config.local.json 的顶层必须是 JSON 对象")
        api_key = config.get("api_key")
        if not isinstance(api_key, str) or not api_key.strip() or api_key == "YOUR_DEEPSEEK_API_KEY":
            raise ValueError("请在 config.local.json 中填写 api_key")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt 必须是字符串")
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.model = config.get("model", "deepseek-flash")
        self.client = AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=config.get("base_url", "https://api.deepseek.com"),
            timeout=30.0,
            max_retries=0,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.client.close()

    async def complete(self, messages: list[Message], tools: list[dict]) -> ModelReply:
        api_messages = [{
            "role": "system",
            "content": self.system_prompt,
        }]
        for message in messages:
            if message.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    } for call in message.tool_calls],
                })
            elif message.role == "tool":
                if message.result is None:
                    raise ValueError("tool 消息缺少 ToolResult")
                result = message.result
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": json.dumps({
                        "ok": not result.is_error,
                        "content": result.content,
                    }, ensure_ascii=False),
                })
            else:
                api_messages.append({"role": message.role, "content": message.content})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                tools=[{"type": "function", "function": tool} for tool in tools],
                stream=False,
                max_tokens=1024,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APITimeoutError:
            raise RuntimeError("DeepSeek 请求超时，请检查网络后重试") from None
        except APIConnectionError:
            raise RuntimeError("无法连接 DeepSeek，请检查网络和代理设置") from None
        except APIStatusError as exc:
            reason = {401: "密钥无效", 402: "余额不足", 429: "请求限流"}.get(
                exc.status_code, "请检查模型、地址、参数或服务状态"
            )
            raise RuntimeError(f"DeepSeek HTTP {exc.status_code}：{reason}") from None

        if not response.choices:
            raise ValueError("DeepSeek 返回了空 choices")
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise ValueError(f"响应未完成：finish_reason={choice.finish_reason}")
        reply = choice.message
        if reply.tool_calls:
            calls = []
            for call in reply.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                except (TypeError, ValueError):
                    raise ValueError("模型返回的工具参数不是合法 JSON") from None
                if not isinstance(arguments, dict):
                    raise ValueError("模型返回的工具参数必须是 JSON 对象")
                calls.append(ToolCall(call.id, call.function.name, arguments))
            # 真实 API 可同时返回说明文字和工具调用，先执行工具。
            return ModelReply(tool_calls=tuple(calls))
        return ModelReply(text=reply.content)
