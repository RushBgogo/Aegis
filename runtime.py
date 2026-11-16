from  __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass,field
from json import tool
from typing import Any,Awaitable,Callable,Protocol
from uuid import uuid4
from pydantic import BaseModel,ConfigDict,ValidationError
from pyexpat.errors import messages

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
    test:str    #最终返回给用户的文本
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
        self.tools: dict[str, ToolSpec]={} #注册表

    def register(self,tool:ToolSpec)->None:
        if tool.name in self._tools:
            raise ValueError(f"重复工具名称:{tool.name}")
        self.tools[tool.name]=tool

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
    )->RunResult:
        if max_turns <1:
            raise ValueError("max_turns必须大于0")
        if events is None:
            events=[]
        if events:
            raise ValueError("events必须是本次运行专用的空列表")
        run_id=uuid4().hex # 生成唯一数值，本轮所有事件共享它
        messages=[Message(role="user",content=prompt)]
        seen_call_ids: set[str] = set() #本来所有出现过的call_id，用于跨轮去重。此为本轮，因为run()开始时候重置该数值，多轮run不受影响
        