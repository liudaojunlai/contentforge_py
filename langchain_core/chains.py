"""
LangChain-style Chain System
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from .messages import HumanMessage, SystemMessage
from .schema import BaseRunnable, RunnableConfig


class BaseChain(BaseRunnable, ABC):
    """Abstract base chain."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    @abstractmethod
    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def ainvoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        return await self._acall(inputs)

    def invoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(inputs, config))

    def __or__(self, other):
        return SequentialChain(chains=[self, other])


class LLMChain(BaseChain):
    """
    Core chain: PromptTemplate → LLM → output.
    LangChain-compatible interface.
    """

    def __init__(
        self,
        llm,
        prompt,
        output_key: str = "output",
        memory=None,
        callbacks=None,
    ):
        super().__init__(callbacks=callbacks)
        self.llm        = llm
        self.prompt     = prompt
        self.output_key = output_key
        self.memory     = memory

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Load memory
        if self.memory:
            mem_vars = self.memory.load_memory_variables()
            inputs = {**inputs, **mem_vars}

        # Format prompt
        if hasattr(self.prompt, 'format_messages'):
            messages = self.prompt.format_messages(**inputs)
        elif hasattr(self.prompt, 'format'):
            text = self.prompt.format(**inputs)
            messages = [HumanMessage(content=text)]
        else:
            messages = [HumanMessage(content=str(inputs))]

        # Separate system message
        system = None
        chat_msgs = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system = m.content
            else:
                chat_msgs.append(m)

        # Call LLM
        result = await self.llm.ainvoke(chat_msgs, system=system)
        output = result.content if hasattr(result, 'content') else str(result)

        # Save to memory
        if self.memory:
            self.memory.save_context(inputs, {self.output_key: output})

        return {self.output_key: output, "_llm_result": result}

    async def apredict(self, **kwargs) -> str:
        result = await self._acall(kwargs)
        return result[self.output_key]

    def predict(self, **kwargs) -> str:
        return asyncio.get_event_loop().run_until_complete(self.apredict(**kwargs))

    @classmethod
    def from_string(cls, llm, template: str, **kwargs) -> "LLMChain":
        from .prompts import PromptTemplate
        prompt = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt, **kwargs)


class SequentialChain(BaseChain):
    """Runs chains sequentially, passing outputs as inputs to the next."""

    def __init__(self, chains: List[BaseChain], input_key: str = "input", callbacks=None):
        super().__init__(callbacks=callbacks)
        self.chains    = chains
        self.input_key = input_key

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        current = dict(inputs)
        all_outputs = {}
        for chain in self.chains:
            result = await chain.ainvoke(current)
            current = {**current, **result}
            all_outputs.update(result)
        return all_outputs

    def __or__(self, other: BaseChain) -> "SequentialChain":
        return SequentialChain(chains=self.chains + [other])


class TransformChain(BaseChain):
    """Applies a pure Python transform function to inputs."""

    def __init__(
        self,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
        callbacks=None,
    ):
        super().__init__(callbacks=callbacks)
        self.transform_fn     = transform_fn
        self.input_variables  = input_variables or []
        self.output_variables = output_variables or []

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if asyncio.iscoroutinefunction(self.transform_fn):
            return await self.transform_fn(inputs)
        return self.transform_fn(inputs)


class RouterChain(BaseChain):
    """Routes to different chains based on a condition."""

    def __init__(
        self,
        router_fn: Callable[[Dict[str, Any]], str],
        destination_chains: Dict[str, BaseChain],
        default_chain: Optional[BaseChain] = None,
        callbacks=None,
    ):
        super().__init__(callbacks=callbacks)
        self.router_fn          = router_fn
        self.destination_chains = destination_chains
        self.default_chain      = default_chain

    async def _acall(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        route = self.router_fn(inputs)
        chain = self.destination_chains.get(route, self.default_chain)
        if chain is None:
            raise ValueError(f"No chain for route '{route}' and no default chain.")
        return await chain.ainvoke(inputs)
