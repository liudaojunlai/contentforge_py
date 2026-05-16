"""
LangChain-style Memory System
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .messages import BaseMessage, HumanMessage, AIMessage


class BaseMemory(ABC):
    """Abstract base memory."""

    @abstractmethod
    def load_memory_variables(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class ConversationBufferMemory(BaseMemory):
    """Stores the full conversation history as messages."""

    def __init__(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "history",
        return_messages: bool = True,
        max_messages: Optional[int] = None,
    ):
        self.human_prefix   = human_prefix
        self.ai_prefix      = ai_prefix
        self.memory_key     = memory_key
        self.return_messages = return_messages
        self.max_messages   = max_messages
        self.messages: List[BaseMessage] = []

    def load_memory_variables(self) -> Dict[str, Any]:
        msgs = self.messages
        if self.max_messages:
            msgs = msgs[-self.max_messages:]

        if self.return_messages:
            return {self.memory_key: msgs}
        else:
            lines = []
            for m in msgs:
                prefix = self.human_prefix if isinstance(m, HumanMessage) else self.ai_prefix
                lines.append(f"{prefix}: {m.content}")
            return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        human_text = inputs.get("input") or inputs.get("question") or str(inputs)
        ai_text    = outputs.get("output") or outputs.get("answer") or str(outputs)
        self.messages.append(HumanMessage(content=human_text))
        self.messages.append(AIMessage(content=ai_text))

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))

    def clear(self) -> None:
        self.messages = []

    @property
    def buffer(self) -> List[BaseMessage]:
        return self.messages

    def __len__(self) -> int:
        return len(self.messages)


class SummaryMemory(BaseMemory):
    """Summarizes old messages to save context window space."""

    def __init__(self, llm=None, memory_key: str = "summary"):
        self.llm = llm
        self.memory_key = memory_key
        self.summary: str = ""
        self.recent_messages: List[BaseMessage] = []
        self.max_recent = 6

    def load_memory_variables(self) -> Dict[str, Any]:
        return {
            self.memory_key: self.summary,
            "recent_messages": self.recent_messages,
        }

    def save_context(self, inputs: Dict, outputs: Dict) -> None:
        human_text = inputs.get("input", "")
        ai_text    = outputs.get("output", "")
        self.recent_messages.append(HumanMessage(content=human_text))
        self.recent_messages.append(AIMessage(content=ai_text))

        if len(self.recent_messages) > self.max_recent:
            old = self.recent_messages[:-self.max_recent]
            self.recent_messages = self.recent_messages[-self.max_recent:]
            old_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in old
            )
            self.summary = f"{self.summary}\n{old_text}".strip()

    def clear(self) -> None:
        self.summary = ""
        self.recent_messages = []
