"""
LangChain-style Prompt Templates
"""
from typing import List, Dict, Any, Optional, Union
from .messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
import re


class PromptTemplate:
    """Simple f-string style prompt template with {variable} substitution."""

    def __init__(self, template: str, input_variables: Optional[List[str]] = None):
        self.template = template
        self.input_variables = input_variables or self._extract_variables(template)

    def _extract_variables(self, template: str) -> List[str]:
        return list(set(re.findall(r'\{(\w+)\}', template)))

    def format(self, **kwargs) -> str:
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables for prompt: {missing}")
        return self.template.format(**kwargs)

    def format_messages(self, **kwargs) -> List[HumanMessage]:
        return [HumanMessage(content=self.format(**kwargs))]

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template=template)

    def __or__(self, other):
        """Support pipe operator: template | llm"""
        from .chains import LLMChain
        if hasattr(other, 'invoke'):
            return _PipelineChain(self, other)
        raise TypeError(f"Cannot pipe PromptTemplate with {type(other)}")


class MessagePlaceholder:
    """Placeholder for inserting a list of messages."""
    def __init__(self, variable_name: str):
        self.variable_name = variable_name


class ChatPromptTemplate:
    """Template for building chat message lists."""

    def __init__(self, messages_template: List[Union[tuple, MessagePlaceholder]]):
        self.messages_template = messages_template
        self.input_variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        vars_ = set()
        for item in self.messages_template:
            if isinstance(item, MessagePlaceholder):
                vars_.add(item.variable_name)
            elif isinstance(item, tuple):
                _, content = item
                vars_.update(re.findall(r'\{(\w+)\}', content))
        return list(vars_)

    def format_messages(self, **kwargs) -> List[BaseMessage]:
        result = []
        for item in self.messages_template:
            if isinstance(item, MessagePlaceholder):
                msgs = kwargs.get(item.variable_name, [])
                result.extend(msgs if isinstance(msgs, list) else [msgs])
            elif isinstance(item, tuple):
                role, template = item
                content = template.format(**{k: v for k, v in kwargs.items()
                                            if re.search(r'\{' + k + r'\}', template)})
                if role == "system":
                    result.append(SystemMessage(content=content))
                elif role == "human":
                    result.append(HumanMessage(content=content))
                elif role == "ai":
                    result.append(AIMessage(content=content))
        return result

    @classmethod
    def from_messages(cls, messages: List) -> "ChatPromptTemplate":
        return cls(messages_template=messages)

    def __or__(self, other):
        from .chains import LLMChain
        return _PipelineChain(self, other)


class _PipelineChain:
    """Internal: supports the | pipe operator between components."""
    def __init__(self, *components):
        self.components = list(components)

    def __or__(self, other):
        return _PipelineChain(*self.components, other)

    async def ainvoke(self, inputs: Dict[str, Any]) -> Any:
        current = inputs
        for comp in self.components:
            if hasattr(comp, 'format_messages'):
                current = comp.format_messages(**current) if isinstance(current, dict) else current
            elif hasattr(comp, 'ainvoke'):
                current = await comp.ainvoke(current)
            elif hasattr(comp, 'invoke'):
                current = comp.invoke(current)
        return current

    def invoke(self, inputs: Dict[str, Any]) -> Any:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(inputs))
