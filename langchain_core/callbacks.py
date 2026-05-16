"""
LangChain-style Callback System & Event Bus
"""
import asyncio
from abc import ABC
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field


class EventType(str, Enum):
    # LLM events
    LLM_START     = "llm_start"
    LLM_END       = "llm_end"
    LLM_ERROR     = "llm_error"
    # Chain events
    CHAIN_START   = "chain_start"
    CHAIN_END     = "chain_end"
    CHAIN_ERROR   = "chain_error"
    # Agent events
    AGENT_ACTION  = "agent_action"
    AGENT_FINISH  = "agent_finish"
    AGENT_ERROR   = "agent_error"
    AGENT_RETRY   = "agent_retry"
    # Tool events
    TOOL_START    = "tool_start"
    TOOL_END      = "tool_end"
    TOOL_ERROR    = "tool_error"
    # Harness events
    PIPELINE_START = "pipeline_start"
    PIPELINE_END   = "pipeline_end"
    STEP_START     = "step_start"
    STEP_END       = "step_end"
    # Skill events
    SKILL_START   = "skill_start"
    SKILL_END     = "skill_end"
    SKILL_ERROR   = "skill_error"
    # General
    LOG           = "log"
    CONTEXT_UPDATE = "context_update"


@dataclass
class CallbackEvent:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.data.get(key, default)


class BaseCallbackHandler(ABC):
    """Base class for all callback handlers."""

    def on_event(self, event: CallbackEvent) -> None:
        """Called for every event. Override for custom handling."""
        handler = getattr(self, f"on_{event.type.value.replace(':', '_')}", None)
        if handler:
            handler(event)

    # Convenience methods - override as needed
    def on_llm_start(self, event: CallbackEvent): pass
    def on_llm_end(self, event: CallbackEvent): pass
    def on_chain_start(self, event: CallbackEvent): pass
    def on_chain_end(self, event: CallbackEvent): pass
    def on_agent_action(self, event: CallbackEvent): pass
    def on_agent_finish(self, event: CallbackEvent): pass
    def on_tool_start(self, event: CallbackEvent): pass
    def on_tool_end(self, event: CallbackEvent): pass
    def on_log(self, event: CallbackEvent): pass
    def on_pipeline_start(self, event: CallbackEvent): pass
    def on_pipeline_end(self, event: CallbackEvent): pass
    def on_step_start(self, event: CallbackEvent): pass
    def on_step_end(self, event: CallbackEvent): pass


class CallbackManager:
    """Manages multiple callback handlers."""

    def __init__(self, handlers: Optional[List[BaseCallbackHandler]] = None):
        self.handlers: List[BaseCallbackHandler] = handlers or []

    def add_handler(self, handler: BaseCallbackHandler) -> None:
        self.handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        self.handlers = [h for h in self.handlers if h is not handler]

    def emit(self, event_type: EventType, **data) -> None:
        event = CallbackEvent(type=event_type, data=data)
        for h in self.handlers:
            try:
                h.on_event(event)
            except Exception:
                pass  # Never let callback errors break the pipeline

    async def aemit(self, event_type: EventType, **data) -> None:
        self.emit(event_type, **data)


class EventBus(CallbackManager):
    """
    Global event bus supporting both handler objects and raw listener functions.
    Singleton pattern for harness-level events.
    """

    def __init__(self):
        super().__init__()
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, fn: Callable) -> Callable:
        """Register a listener function. Returns unsubscribe fn."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(fn)
        def unsubscribe():
            self._listeners[event_type] = [
                f for f in self._listeners[event_type] if f is not fn
            ]
        return unsubscribe

    def emit(self, event_type: EventType, **data) -> None:
        super().emit(event_type, **data)
        # Also call raw listeners
        key = event_type.value if isinstance(event_type, EventType) else event_type
        for fn in self._listeners.get(key, []):
            try:
                fn(CallbackEvent(type=event_type, data=data))
            except Exception:
                pass
        # Wildcard listeners
        for fn in self._listeners.get("*", []):
            try:
                fn(CallbackEvent(type=event_type, data=data))
            except Exception:
                pass


# ─────────────────────────────────────────────
# Built-in: Rich Console Logger
# ─────────────────────────────────────────────
class ConsoleCallbackHandler(BaseCallbackHandler):
    """Prints colored pipeline events to the console."""

    COLORS = {
        "reset":  "\033[0m",
        "bold":   "\033[1m",
        "dim":    "\033[2m",
        "purple": "\033[35m",
        "cyan":   "\033[36m",
        "green":  "\033[32m",
        "yellow": "\033[33m",
        "red":    "\033[31m",
        "blue":   "\033[34m",
    }

    def _c(self, text: str, *colors: str) -> str:
        codes = "".join(self.COLORS.get(c, "") for c in colors)
        return f"{codes}{text}{self.COLORS['reset']}"

    def on_event(self, event: CallbackEvent) -> None:
        et = event.type
        d  = event.data
        if et == EventType.PIPELINE_START:
            print(self._c(f"\n🚀 Pipeline 启动: {d.get('name','')}", "bold", "purple"))
        elif et == EventType.STEP_START:
            print(self._c(f"  ▶ [{d.get('step_id','')}] {d.get('name','')} 开始...", "cyan"))
        elif et == EventType.STEP_END:
            print(self._c(f"  ✓ [{d.get('step_id','')}] 完成", "green"))
        elif et == EventType.AGENT_ERROR:
            print(self._c(f"  ✗ 错误: {d.get('error','')}", "red"))
        elif et == EventType.AGENT_RETRY:
            print(self._c(f"  ↻ 重试 ({d.get('attempt',1)})", "yellow"))
        elif et == EventType.TOOL_START:
            print(self._c(f"    🔧 Tool [{d.get('tool','')}] 调用中...", "dim"))
        elif et == EventType.TOOL_END:
            print(self._c(f"    ✓ Tool 完成", "dim", "green"))
        elif et == EventType.SKILL_START:
            print(self._c(f"    💡 Skill [{d.get('skill','')}] 激活...", "blue"))
        elif et == EventType.SKILL_END:
            print(self._c(f"    ✓ Skill 完成", "blue"))
        elif et == EventType.PIPELINE_END:
            print(self._c(f"\n✅ Pipeline 完成！耗时 {d.get('duration',0):.1f}s\n", "bold", "green"))
        elif et == EventType.LOG:
            level = d.get("level", "info")
            color = {"info": "dim", "warn": "yellow", "error": "red"}.get(level, "dim")
            print(self._c(f"  ℹ {d.get('message','')}", color))
