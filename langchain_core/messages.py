"""
LangChain-style Message Types
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM    = "system"
    HUMAN     = "user"
    AI        = "assistant"
    TOOL      = "tool"
    FUNCTION  = "function"


@dataclass
class BaseMessage:
    content: str
    role: MessageRole = MessageRole.HUMAN
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"{self.__class__.__name__}('{preview}...')"


@dataclass
class HumanMessage(BaseMessage):
    role: MessageRole = field(default=MessageRole.HUMAN, init=False)

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, role=MessageRole.HUMAN, **kwargs)


@dataclass
class AIMessage(BaseMessage):
    role: MessageRole = field(default=MessageRole.AI, init=False)
    tool_calls: List[Dict] = field(default_factory=list)

    def __init__(self, content: str, tool_calls: Optional[List] = None, **kwargs):
        super().__init__(content=content, role=MessageRole.AI, **kwargs)
        self.tool_calls = tool_calls or []


@dataclass
class SystemMessage(BaseMessage):
    role: MessageRole = field(default=MessageRole.SYSTEM, init=False)

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, role=MessageRole.SYSTEM, **kwargs)


@dataclass
class ToolMessage(BaseMessage):
    role: MessageRole = field(default=MessageRole.TOOL, init=False)
    tool_name: str = ""

    def __init__(self, content: str, tool_name: str = "", **kwargs):
        super().__init__(content=content, role=MessageRole.TOOL, **kwargs)
        self.tool_name = tool_name


def messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Convert list of messages to API-compatible dicts."""
    return [m.to_dict() for m in messages]
