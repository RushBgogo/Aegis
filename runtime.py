from  __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass,field
from typing import Any,Awaitable,Callable,Protocol
from uuid import uuid4
from pydantic import BaseModel,ConfigDict,ValidationError

'''
实现执行框架
'''


#实例不可变，当工具一旦产生就当作事实记录下来，后面无论加到message还是emit到事件流，都不应该随意改变
@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class ToolResult:
    call_id: str   #和ToolCall匹配
    name: str
    content: Any  #返回的信息可以是任意数值
    is_error: bool=False

#这里toolcalls用元组，是因为frozen只能繁殖给属性重新复制，防不住修改对象内部
@dataclass(frozen=True)
class Message:
    role:str
    content:str=""  #带有默认数值
    tool_calls:tuple[ToolCall,...]=() #元素全是T、长度任意的元组
    result:ToolResult | None=None   #表示ToolResult 或 None

@dataclass(frozen=True)
class ModelReply:
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


#如果携程data:dict={}，所有实例都会共用同一个字典，这里每个实例保存自己的字典
@dataclass(frozen=True)
class RuntimeEvent:
    seq:int
    run_id:str   #本次运行ID
    kind:str     #事件类型
    call_id:str | None=None
    data:dict[str, Any] =field(default_factory=dict)

@dataclass(frozen=True)
class RunResult:
    text:str    #最终返回给用户的文本
    reason:str  #结束原因
    turns:int   #用来多少轮
    messages:list[Message]  #完整对话历史
    events:list[RuntimeEvent]  #完整的事件流

class ModelClient(Protocol):
    async def complete(
            self,
            messages:list[Message],
            tools:list[dict[str, Any]],
    )->ModelReply:
        ...
'''
以下为工具注册与执行
'''

#工具入参模型基类，其中不做隐式类型转换，防止出现模型输出格式不对强行转换的问题
class ToolInput(BaseModel):
    model_config=ConfigDict(strict=True,extra="forbid")

@dataclass
class ToolSpec:
    name: str   #唯一变量
    description: str  #用于模型选择工具说明书
    input_type:type[ToolInput]  #Pydantic模型类，用于生成JSON Schema给模型，运行时校验入参
    handler: Callable[[Any],Awaitable[Any]]  #接受已经校验ToolInput的实例，返回任意结果

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec]={} #注册表

    def register(self,tool:ToolSpec)->None:
        if tool.name in self._tools:
            raise ValueError(f"重复工具名称:{tool.name}")
        self._tools[tool.name]=tool

    #把注册表转换成模型可用工具描述,OpenAI/Anthropic常见格式
    def describe(self)->list[dict[str, Any]]:
        return [
            {
                "name":tool.name,
                "description":tool.description,
                "parameters":tool.input_type.model_json_schema(),
            }
            for tool in self._tools.values()
        ]
    async def execute(
            self,
            call:ToolCall,
            emit:Callable[...,None],
    )->ToolResult:
        emit( #只要请求了无论成功都要记录
            "tool_requested",
            call,
            name=call.name,
            arguments=call.arguments,
        )
        def fail(code:str,message:str)->ToolResult:
            error={"code":code,"message":message}
            emit("tool_failed",call,name=call.name,error=error)
            return ToolResult(call.call_id,call.name,error,is_error=True)
        tool=self._tools.get(call.name)
        if tool is None:
            return fail("unknown_tool",f"工具未注册:{call.name}")
        try:
            arguments=tool.input_type.model_validate(call.arguments)
        except ValidationError as exc:
            return fail("invalid_arguments",str(exc))
        emit("tool_started",call,name=call.name)
        try:
            value=await tool.handler(arguments)
        except asyncio.CancelledError:
            emit("tool_cancelled",call,name=call.name)
            raise
        except Exception as exc:
            return fail("tool_error",str(exc))
        emit("tool_finished",call,name=call.name,value=value)
        return ToolResult(call.call_id,call.name,value)

'''
Agent执行循环
'''
class AgentRuntime:
    def __init__(self,model:ModelClient,tools:ToolRegistry):
        self.model=model
        self.tools=tools

    async def run(
            self,
            prompt:str,
            max_turns:int=5,
            events:list[RuntimeEvent] | None=None,
            history:list[Message] | None=None,
    )->RunResult:
        if max_turns <1:
            raise ValueError("max_turns必须大于0")
        if events is None:
            events=[]
        if events:
            raise ValueError("events必须是本次运行专用的空列表")
        run_id=uuid4().hex # 生成唯一数值，本轮所有事件共享它
        # history 不为空时继续已有会话；否则从一条 user 消息开始。
        messages=deepcopy(history) if history is not None else [Message(role="user",content=prompt)]
        if not messages:
            raise ValueError("history cannot be empty")
        seen_call_ids: set[str] = {
            call.call_id
            for message in messages
            for call in message.tool_calls
        } #包括历史中出现过的 call_id，防止恢复会话后重复使用
        #统一事件记录器
        def emit(
                kind:str,
                call:ToolCall | None=None,
                **data:Any,
        )->None:
            events.append(
                RuntimeEvent(
                    seq=len(events)+1,
                    run_id=run_id,
                    kind=kind,
                    call_id=call.call_id if call else None,
                    data=deepcopy(data),
                )
            )
        #统一结束本轮的出口
        def finish(
                reason:str,
                turns:int,
                text:str="",
                error:str="",
        )->RunResult:
            kind=(
                "run_completed"
                if reason =="completed"
                else "run_stopped"
            )
            emit(kind,reason=reason,answer=text,error=error)
            return RunResult(
                text,
                reason,
                turns,
                messages,
                events,
            )
        try:
            #主循环请求模型
            for turn in range(1,max_turns+1):
                emit("model_requested",turn=turn)
                try:
                    reply=await self.model.complete(
                        deepcopy(messages),#提供副本防止模型污染Runtime真实历史，隔离设计
                        self.tools.describe(),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return finish(
                        "model_error",
                        turn,
                        error=str(exc),
                    )
                if not isinstance(reply,ModelReply):
                    return finish(
                        "protocol_error",
                        turn,
                        error="模型返回类型错误",
                    )
                #互斥校验
                if (reply.text is not None) == bool(reply.tool_calls):
                    return finish(
                        "protocol_error",
                        turn,
                        error="文本与工具必须二选一调动",
                    )
                #最终文本
                if reply.text is not None:
                    if(
                        not isinstance(reply.text,str)
                        or not reply.text.strip()
                    ):
                        return finish(
                            "protocol_error",
                            turn,
                            error="最终文本不能为空",
                        )
                    messages.append(
                        Message(
                        role="assistant",
                        content=reply.text,
                        )
                    )
                    return finish(
                        "completed",
                        turn,
                        text=reply.text,
                    )
                #工具调用基础校验
                if any(
                        not isinstance(call, ToolCall)
                        or not isinstance(call.call_id, str)
                        or not call.call_id
                        or not isinstance(call.name, str)
                        or not call.name
                        for call in reply.tool_calls
                ):
                    return finish(
                        "protocol_error",
                        turn,
                        error="工具调用标识无效",
                    )
                ids = [call.call_id for call in reply.tool_calls]

                if (
                        len(ids) != len(set(ids))
                        or seen_call_ids.intersection(ids)
                ):
                    return finish(
                        "protocol_error",
                        turn,
                        error="call_id 重复",
                    )

                seen_call_ids.update(ids)
                messages.append(
                    Message(
                        role="assistant",
                        tool_calls=reply.tool_calls,
                    )
                )

                for call in reply.tool_calls:
                    result = await self.tools.execute(call, emit)

                    messages.append(
                        Message(
                            role="tool",
                            result=result,
                        )
                    )
            return finish("max_turns", max_turns)
        except asyncio.CancelledError:
            emit("run_cancelled")
            raise
