"""
ContentForge LangChain-style Core Framework
============================================
纯 Python 实现，复刻 LangChain 核心抽象：
  - BaseMessage / HumanMessage / AIMessage / SystemMessage
  - PromptTemplate / ChatPromptTemplate
  - BaseLLM / LLMChain
  - BaseMemory / ConversationBufferMemory
  - BaseTool / ToolRegistry
  - BaseAgent / AgentExecutor
  - Callbacks / EventBus
"""

from .messages   import BaseMessage, HumanMessage, AIMessage, SystemMessage
from .prompts    import PromptTemplate, ChatPromptTemplate, MessagePlaceholder
from .llm        import BaseLLM, LLMResult
from .chains     import LLMChain, SequentialChain, TransformChain
from .memory     import BaseMemory, ConversationBufferMemory
from .tools      import BaseTool, ToolResult, ToolRegistry
from .agents     import BaseAgent, AgentExecutor, AgentAction, AgentFinish
from .callbacks  import BaseCallbackHandler, CallbackManager, EventBus
from .schema     import BaseRunnable, RunnableConfig

__all__ = [
    "BaseMessage", "HumanMessage", "AIMessage", "SystemMessage",
    "PromptTemplate", "ChatPromptTemplate", "MessagePlaceholder",
    "BaseLLM", "LLMResult",
    "LLMChain", "SequentialChain", "TransformChain",
    "BaseMemory", "ConversationBufferMemory",
    "BaseTool", "ToolResult", "ToolRegistry",
    "BaseAgent", "AgentExecutor", "AgentAction", "AgentFinish",
    "BaseCallbackHandler", "CallbackManager", "EventBus",
    "BaseRunnable", "RunnableConfig",
]
