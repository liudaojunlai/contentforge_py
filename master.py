"""
ContentForge Master Agent
══════════════════════════════════════════════
主 Agent：接收用户请求，构建 Harness Pipeline，
协调六个子 Agent 完成端到端内容创作。
同时管理 Skill 的挂载与执行。
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .langchain_core.llm import BaseLLM, create_llm
from .harness.engine import Harness, PipelineStep, create_harness
from .harness.skills import SkillRegistry, create_default_skills, BaseSkill
from .agents.content_agents import create_all_agents
from .langchain_core.callbacks import EventBus, EventType


# ─────────────────────────────────────────────
# Pipeline Definition
# ─────────────────────────────────────────────
def build_pipeline(skill_registry: Optional[SkillRegistry] = None) -> List[PipelineStep]:
    """
    Build the content creation pipeline.
    Skills are attached as post-hooks to relevant steps.
    """
    # Writer post-hook: format validation skill
    writer_post_hooks = []
    editor_post_hooks = []

    if skill_registry:
        fmt_skill = skill_registry.get("format_validator")
        wc_skill  = skill_registry.get("word_count")
        rd_skill  = skill_registry.get("readability")
        seo_skill = skill_registry.get("seo_quality")
        tone_skill = skill_registry.get("tone_analysis")

        if wc_skill:
            editor_post_hooks.append(wc_skill.as_post_hook())
        if rd_skill:
            editor_post_hooks.append(rd_skill.as_post_hook())
        if seo_skill:
            editor_post_hooks.append(seo_skill.as_post_hook())
        if tone_skill:
            editor_post_hooks.append(tone_skill.as_post_hook())
        if fmt_skill:
            writer_post_hooks.append(fmt_skill.as_post_hook())

    return [
        PipelineStep(
            agent_id="topic",
            output_key="topicResult",
            name="选题策划",
            description="分析需求，确定最优写作角度",
            retries=2,
            timeout=60.0,
        ),
        PipelineStep(
            agent_id="research",
            output_key="researchResult",
            name="调研分析",
            description="收集背景知识、数据与案例",
            retries=2,
            timeout=60.0,
        ),
        PipelineStep(
            agent_id="outline",
            output_key="outlineResult",
            name="结构设计",
            description="规划文章骨架与章节逻辑",
            retries=2,
            timeout=60.0,
        ),
        PipelineStep(
            agent_id="writer",
            output_key="writerResult",
            name="内容撰写",
            description="基于大纲和素材生成完整正文",
            retries=1,
            timeout=120.0,
            post_hooks=writer_post_hooks,
        ),
        PipelineStep(
            agent_id="editor",
            output_key="editorResult",
            name="编辑润色",
            description="优化语言表达与逻辑结构",
            retries=1,
            timeout=120.0,
            post_hooks=editor_post_hooks,
        ),
        PipelineStep(
            agent_id="seo",
            output_key="seoResult",
            name="SEO 分析",
            description="提取关键词，生成 Meta 信息",
            retries=2,
            timeout=60.0,
        ),
    ]


# ─────────────────────────────────────────────
# Content Request / Response
# ─────────────────────────────────────────────
@dataclass
class ContentRequest:
    user_request: str
    platform: str    = "通用媒体"
    style: str       = "专业干货"
    word_count: int  = 1500

    def to_context(self) -> Dict[str, Any]:
        return {
            "userRequest": self.user_request,
            "platform":    self.platform,
            "style":       self.style,
            "wordCount":   self.word_count,
        }


@dataclass
class ContentResult:
    topic:    Dict[str, Any]  = field(default_factory=dict)
    research: Dict[str, Any]  = field(default_factory=dict)
    outline:  Dict[str, Any]  = field(default_factory=dict)
    article:  str             = ""
    seo:      Dict[str, Any]  = field(default_factory=dict)
    skills:   Dict[str, Any]  = field(default_factory=dict)
    duration: float           = 0.0
    logs:     List[Dict]      = field(default_factory=list)
    agent_statuses: List[Dict] = field(default_factory=list)
    success:  bool            = True
    error:    Optional[str]   = None


# ─────────────────────────────────────────────
# Master Agent
# ─────────────────────────────────────────────
class MasterAgent:
    """
    Top-level orchestrator.
    Wires together: LLM → Sub-Agents → Harness → Skills → Output
    """

    def __init__(
        self,
        llm: BaseLLM,
        skill_registry: Optional[SkillRegistry] = None,
        on_event: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.llm      = llm
        self.skills   = skill_registry or create_default_skills()
        self.harness  = create_harness(name="ContentForge", verbose=verbose)
        self._on_event = on_event

        # Register sub-agents
        for agent in create_all_agents(llm):
            self.harness.register(agent)

        # Forward events to external listener
        if on_event:
            self.harness.bus.on("*", on_event)

    # ── Mount custom skill ──────────────────
    def add_skill(self, skill: BaseSkill) -> "MasterAgent":
        """Dynamically add a skill."""
        self.skills.register(skill)
        return self

    # ── Execute pipeline ────────────────────
    async def acreate(self, request: ContentRequest) -> ContentResult:
        start = time.time()
        self.harness.reset()

        pipeline = build_pipeline(self.skills)

        try:
            await self.harness.run(
                steps=pipeline,
                initial_context=request.to_context(),
            )
            ctx = self.harness.context

            # Collect skill results
            skill_data = {}
            for key in ctx.all():
                if key.startswith("skill_"):
                    skill_data[key[6:]] = ctx.get(key)

            # Agent status summary
            statuses = [
                {
                    "id":     a.id,
                    "name":   a.name,
                    "icon":   a.icon,
                    "color":  a.color,
                    "status": a.status,
                }
                for a in self.harness.registry.all()
            ]

            return ContentResult(
                topic    = ctx.get("topicResult")    or {},
                research = ctx.get("researchResult") or {},
                outline  = ctx.get("outlineResult")  or {},
                article  = ctx.get("editorResult")   or ctx.get("writerResult") or "",
                seo      = ctx.get("seoResult")      or {},
                skills   = skill_data,
                duration = time.time() - start,
                logs     = [e.to_dict() for e in self.harness.logs],
                agent_statuses = statuses,
                success  = True,
            )

        except Exception as e:
            return ContentResult(
                success  = False,
                error    = str(e),
                duration = time.time() - start,
                logs     = [e_log.to_dict() for e_log in self.harness.logs],
            )

    def create(self, request: ContentRequest) -> ContentResult:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.acreate(request))
        finally:
            loop.close()

    @property
    def agents(self):
        return self.harness.registry.all()


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def create_master(
    provider: str,
    api_key: str,
    model: str = "",
    base_url: str = "",
    skills: Optional[List[BaseSkill]] = None,
    verbose: bool = True,
    **llm_kwargs,
) -> MasterAgent:
    """
    ════════════════════════════════════════
    API 接口入口 — 填入你的大模型 API Key
    ════════════════════════════════════════
    """
    llm = create_llm(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        **llm_kwargs,
    )

    skill_registry = create_default_skills()
    if skills:
        for s in skills:
            skill_registry.register(s)

    return MasterAgent(llm=llm, skill_registry=skill_registry, verbose=verbose)
