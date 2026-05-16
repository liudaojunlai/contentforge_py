"""
LangChain-style Tool System
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Type
import json


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    output: Any
    tool_name: str
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.success:
            return f"[Tool Error] {self.error}"
        if isinstance(self.output, (dict, list)):
            return json.dumps(self.output, ensure_ascii=False, indent=2)
        return str(self.output)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools can be:
    - Standalone utilities (web search, calculator)
    - Skill wrappers (SEO checker, readability scorer)
    - Sub-agent callers
    """

    name: str = "base_tool"
    description: str = "A base tool"
    args_schema: Optional[Dict] = None   # JSON Schema for args
    return_direct: bool = False          # If True, return output directly without further processing

    @abstractmethod
    async def _arun(self, **kwargs) -> Any:
        """Async execution. Implement this."""
        pass

    def _run(self, **kwargs) -> Any:
        """Sync execution. Wraps async."""
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))

    async def ainvoke(self, input: Any, **kwargs) -> ToolResult:
        """Invoke the tool with input (dict or string)."""
        try:
            if isinstance(input, dict):
                output = await self._arun(**input)
            elif isinstance(input, str):
                output = await self._arun(input=input)
            else:
                output = await self._arun(input=input)
            return ToolResult(output=output, tool_name=self.name, success=True)
        except Exception as e:
            return ToolResult(output=None, tool_name=self.name, success=False, error=str(e))

    def invoke(self, input: Any, **kwargs) -> ToolResult:
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(input, **kwargs))

    def as_schema(self) -> Dict:
        """Return tool schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema or {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Tool input"}
                }
            }
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def register_many(self, tools: List[BaseTool]) -> "ToolRegistry":
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list(self) -> List[str]:
        return list(self._tools.keys())

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict]:
        return [t.as_schema() for t in self._tools.values()]

    async def ainvoke(self, name: str, input: Any) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(output=None, tool_name=name, success=False,
                            error=f"Tool '{name}' not found. Available: {self.list()}")
        return await tool.ainvoke(input)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list()})"
