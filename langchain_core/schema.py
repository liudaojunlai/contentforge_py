"""
Base Schema & Runnable Interface
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RunnableConfig:
    tags: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    callbacks: list = field(default_factory=list)
    max_concurrency: int = 4
    run_name: Optional[str] = None


class BaseRunnable(ABC):
    """Base interface for all runnable components."""

    @abstractmethod
    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        pass

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, config))

    def __or__(self, other: "BaseRunnable") -> "BaseRunnable":
        from .chains import SequentialChain
        return SequentialChain(chains=[self, other])
