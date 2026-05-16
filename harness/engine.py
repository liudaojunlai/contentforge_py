"""
ContentForge Harness Engine
════════════════════════════════════════════════
Harness 是系统的核心调度框架，基于 LangChain 抽象构建。
提供：
  - Agent 注册中心（AgentRegistry）
  - Pipeline 定义（PipelineStep）
  - 上下文总线（ContextBus）
  - 错误重试 & 超时控制
  - 全链路事件追踪
  - Skill 挂载点
"""
import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type
from enum import Enum

from ..langchain_core.callbacks import EventBus, EventType, ConsoleCallbackHandler
from ..langchain_core.agents import BaseAgent

logger = logging.getLogger("contentforge.harness")


# ─────────────────────────────────────────────
# Context Bus
# ─────────────────────────────────────────────
class ContextBus:
    """
    Shared state store passed through all pipeline steps.
    Acts as LangChain's 'memory' at the pipeline level.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        self._store: Dict[str, Any] = initial or {}
        self._history: List[Dict] = []

    def set(self, key: str, value: Any, agent_id: str = "") -> None:
        self._store[key] = value
        self._history.append({
            "key": key,
            "agent": agent_id,
            "ts": time.time(),
            "type": type(value).__name__,
        })

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def all(self) -> Dict[str, Any]:
        return dict(self._store)

    def snapshot(self) -> Dict[str, Any]:
        """Deep copy of current context."""
        try:
            return json.loads(json.dumps(self._store, default=str))
        except Exception:
            return dict(self._store)

    def reset(self) -> None:
        self._store = {}
        self._history = []

    def update(self, data: Dict[str, Any], agent_id: str = "") -> None:
        for k, v in data.items():
            self.set(k, v, agent_id)

    @property
    def history(self) -> List[Dict]:
        return list(self._history)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"ContextBus(keys={keys})"


# ─────────────────────────────────────────────
# Pipeline Step Definition
# ─────────────────────────────────────────────
@dataclass
class PipelineStep:
    """
    Defines one step in the Harness pipeline.
    Analogous to a LangChain chain step.
    """
    agent_id: str
    output_key: str                        # Where to store output in ContextBus
    name: str = ""
    description: str = ""
    retries: int = 2
    timeout: float = 90.0                  # seconds
    condition: Optional[Callable[[ContextBus], bool]] = None  # Skip if False
    pre_hooks: List[Callable] = field(default_factory=list)   # Before agent runs
    post_hooks: List[Callable] = field(default_factory=list)  # After agent runs
    parallel_group: Optional[str] = None  # Steps in same group run concurrently
    required: bool = True                  # If False, failure doesn't stop pipeline

    def should_run(self, context: ContextBus) -> bool:
        if self.condition is None:
            return True
        return bool(self.condition(context))


# ─────────────────────────────────────────────
# Agent Registry
# ─────────────────────────────────────────────
class AgentRegistry:
    """Central registry for all agents in the harness."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._metadata: Dict[str, Dict] = {}

    def register(self, agent: BaseAgent, tags: Optional[List[str]] = None) -> "AgentRegistry":
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Expected BaseAgent, got {type(agent)}")
        self._agents[agent.id] = agent
        self._metadata[agent.id] = {
            "tags": tags or [],
            "registered_at": time.time(),
        }
        return self

    def get(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def require(self, agent_id: str) -> BaseAgent:
        agent = self.get(agent_id)
        if not agent:
            raise KeyError(f"Agent '{agent_id}' not found. Registered: {self.list()}")
        return agent

    def list(self) -> List[str]:
        return list(self._agents.keys())

    def all(self) -> List[BaseAgent]:
        return list(self._agents.values())

    def by_tag(self, tag: str) -> List[BaseAgent]:
        return [
            self._agents[aid]
            for aid, meta in self._metadata.items()
            if tag in meta.get("tags", [])
        ]

    def reset_all(self) -> None:
        for agent in self._agents.values():
            agent.reset()

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={self.list()})"


# ─────────────────────────────────────────────
# Execution Log Entry
# ─────────────────────────────────────────────
@dataclass
class LogEntry:
    timestamp: float
    level: str       # info | warn | error
    message: str
    agent_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "ts": self.timestamp,
            "level": self.level,
            "message": self.message,
            "agent_id": self.agent_id,
        }


# ─────────────────────────────────────────────
# Harness Engine
# ─────────────────────────────────────────────
class Harness:
    """
    Core orchestration engine.
    Registers agents, defines pipelines, manages context and events.
    """

    def __init__(self, name: str = "contentforge", verbose: bool = True):
        self.name     = name
        self.registry = AgentRegistry()
        self.context  = ContextBus()
        self.bus      = EventBus()
        self._logs: List[LogEntry] = []
        self._start_time: Optional[float] = None

        if verbose:
            self.bus.add_handler(ConsoleCallbackHandler())

    # ── Agent Management ────────────────────
    def register(self, agent: BaseAgent, tags: Optional[List[str]] = None) -> "Harness":
        self.registry.register(agent, tags=tags)
        self._log(f"Registered agent: [{agent.id}] {agent.name}", "info")
        return self

    def agent(self, agent_id: str) -> BaseAgent:
        return self.registry.require(agent_id)

    # ── Logging ─────────────────────────────
    def _log(self, message: str, level: str = "info", agent_id: str = "") -> None:
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            agent_id=agent_id,
        )
        self._logs.append(entry)
        self.bus.emit(EventType.LOG, level=level, message=message, agent_id=agent_id)

    @property
    def logs(self) -> List[LogEntry]:
        return list(self._logs)

    # ── Run single agent (with retry + timeout) ──
    async def _run_step(self, step: PipelineStep) -> Any:
        agent = self.registry.require(step.agent_id)
        step_name = step.name or agent.name

        # Check condition
        if not step.should_run(self.context):
            self._log(f"Skipping [{step.agent_id}] (condition not met)", "warn")
            return None

        # Pre-hooks
        for hook in step.pre_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(self.context)
            else:
                hook(self.context)

        self.bus.emit(EventType.STEP_START,
                      step_id=step.agent_id, name=step_name,
                      description=step.description)

        last_error = None
        for attempt in range(step.retries + 1):
            if attempt > 0:
                self.bus.emit(EventType.AGENT_RETRY,
                              agent_id=step.agent_id, attempt=attempt)
                await asyncio.sleep(1.5 * attempt)

            try:
                ctx = self.context.all()
                # Timeout wrapper
                output = await asyncio.wait_for(
                    agent.execute(ctx),
                    timeout=step.timeout,
                )
                # Store in context
                if step.output_key:
                    self.context.set(step.output_key, output, agent_id=step.agent_id)

                # Post-hooks
                for hook in step.post_hooks:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(self.context, output)
                    else:
                        hook(self.context, output)

                self.bus.emit(EventType.STEP_END,
                              step_id=step.agent_id, name=step_name,
                              output_key=step.output_key)
                self._log(f"[{step.agent_id}] completed", "info", agent_id=step.agent_id)
                return output

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Agent [{step.agent_id}] timed out ({step.timeout}s)")
                self._log(str(last_error), "error", agent_id=step.agent_id)
                self.bus.emit(EventType.AGENT_ERROR,
                              agent_id=step.agent_id, error=str(last_error), attempt=attempt)
            except Exception as e:
                last_error = e
                self._log(f"[{step.agent_id}] error: {e}", "error", agent_id=step.agent_id)
                self.bus.emit(EventType.AGENT_ERROR,
                              agent_id=step.agent_id, error=str(e), attempt=attempt)

        # All retries exhausted
        if step.required:
            raise last_error or RuntimeError(f"Step [{step.agent_id}] failed")
        else:
            self._log(f"[{step.agent_id}] failed (non-required, continuing)", "warn")
            return None

    # ── Run Pipeline ─────────────────────────
    async def run(
        self,
        steps: List[PipelineStep],
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full pipeline."""
        self._start_time = time.time()
        self._logs = []

        # Validate all agents are registered
        for step in steps:
            if step.agent_id not in self.registry:
                raise KeyError(f"Agent '{step.agent_id}' not registered. Call harness.register() first.")

        # Seed context
        if initial_context:
            self.context.update(initial_context)

        # Reset agents
        self.registry.reset_all()

        self.bus.emit(EventType.PIPELINE_START,
                      name=self.name,
                      steps=[s.agent_id for s in steps])
        self._log(f"Pipeline '{self.name}' started ({len(steps)} steps)", "info")

        results: Dict[str, Any] = {}

        # Group parallel steps
        i = 0
        while i < len(steps):
            step = steps[i]

            # Check if this starts a parallel group
            if step.parallel_group:
                group_id = step.parallel_group
                group_steps = [s for s in steps[i:] if s.parallel_group == group_id]
                self._log(f"Running {len(group_steps)} steps in parallel (group: {group_id})", "info")
                outputs = await asyncio.gather(
                    *[self._run_step(s) for s in group_steps],
                    return_exceptions=True,
                )
                for s, out in zip(group_steps, outputs):
                    if isinstance(out, Exception):
                        if s.required:
                            raise out
                    else:
                        results[s.agent_id] = out
                i += len(group_steps)
            else:
                output = await self._run_step(step)
                results[step.agent_id] = output
                i += 1

        duration = time.time() - self._start_time
        self.bus.emit(EventType.PIPELINE_END,
                      name=self.name,
                      duration=duration,
                      results=list(results.keys()))
        self._log(f"Pipeline completed in {duration:.1f}s", "info")

        return results

    # ── Convenience: sync run ────────────────
    def run_sync(
        self,
        steps: List[PipelineStep],
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run(steps, initial_context))
        finally:
            loop.close()

    # ── Reset ────────────────────────────────
    def reset(self) -> None:
        self.context.reset()
        self.registry.reset_all()
        self._logs = []

    def __repr__(self) -> str:
        return f"Harness(name='{self.name}', agents={self.registry.list()})"


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def create_harness(name: str = "contentforge", verbose: bool = True) -> Harness:
    """Create a new Harness instance."""
    return Harness(name=name, verbose=verbose)
