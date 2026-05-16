"""
LangChain-style Agent System
"""
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from .messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from .tools import BaseTool, ToolRegistry, ToolResult
from .schema import BaseRunnable


@dataclass
class AgentAction:
    """Represents an action the agent wants to take."""
    tool: str
    tool_input: Dict[str, Any]
    log: str = ""
    thought: str = ""


@dataclass
class AgentFinish:
    """Represents the agent's final output."""
    output: Any
    log: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


AgentStep = Union[AgentAction, AgentFinish]


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Harness framework.
    An Agent = LLM + Tools + Prompt + (optional) Skills
    """

    id: str = "base_agent"
    name: str = "Base Agent"
    description: str = ""
    icon: str = "🤖"
    color: str = "#6366f1"

    def __init__(
        self,
        llm=None,
        tools: Optional[List[BaseTool]] = None,
        memory=None,
        callbacks=None,
        max_iterations: int = 5,
        **kwargs
    ):
        self.llm            = llm
        self.tool_registry  = ToolRegistry()
        self.memory         = memory
        self.callbacks      = callbacks or []
        self.max_iterations = max_iterations
        self._status        = "idle"  # idle | running | done | error
        self._output        = None
        self._error         = None

        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

    @property
    def status(self) -> str:
        return self._status

    @property
    def output(self) -> Any:
        return self._output

    def add_tool(self, tool: BaseTool) -> "BaseAgent":
        self.tool_registry.register(tool)
        return self

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    async def build_messages(self, context: Dict[str, Any]) -> List[BaseMessage]:
        """Build the messages to send to the LLM."""
        pass

    def post_process(self, raw_output: str, context: Dict[str, Any]) -> Any:
        """Optional: post-process LLM output. Override as needed."""
        return raw_output

    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the agent. Called by the Harness."""
        self._status = "running"
        self._output = None
        self._error  = None

        messages = await self.build_messages(context)
        system   = self.get_system_prompt()

        result = await self.llm.ainvoke(messages, system=system)
        raw    = result.content if hasattr(result, 'content') else str(result)

        self._output = self.post_process(raw, context)
        self._status = "done"
        return self._output

    def reset(self) -> None:
        self._status = "idle"
        self._output = None
        self._error  = None


class AgentExecutor(BaseRunnable):
    """
    Runs an agent in a ReAct-style loop:
    Thought → Action → Observation → ... → Final Answer
    """

    def __init__(
        self,
        agent: BaseAgent,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 5,
        early_stopping_method: str = "force",
        callbacks=None,
    ):
        self.agent = agent
        self.max_iterations = max_iterations
        self.early_stopping_method = early_stopping_method
        self.callbacks = callbacks or []

        if tools:
            for tool in tools:
                self.agent.add_tool(tool)

    async def ainvoke(self, inputs: Dict[str, Any], config=None) -> Dict[str, Any]:
        context = dict(inputs)
        output  = await self.agent.execute(context)
        return {"output": output, "agent_id": self.agent.id}

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: BaseAgent,
        tools: List[BaseTool],
        **kwargs
    ) -> "AgentExecutor":
        return cls(agent=agent, tools=tools, **kwargs)
