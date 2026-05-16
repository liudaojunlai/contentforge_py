"""
ContentForge Skill System
═══════════════════════════════════════════════
Skill 是可插拔的功能模块，可以挂载到任意 Agent 上。
与 LangChain Tool 的区别：
  - Tool = 原子操作（搜索、计算、API调用）
  - Skill = 面向业务的增强能力，可组合多个 Tool，
            可以在 Agent 的 pre/post process 阶段注入

内置 Skill:
  - SEOSkill          — SEO 质量评分与关键词建议
  - ReadabilitySkill  — 可读性分析（Flesch-like）
  - FactCheckSkill    — 基本事实一致性检查
  - FormatSkill       — 输出格式标准化
  - WordCountSkill    — 字数统计与段落分析
  - ToneAnalysisSkill — 语气分析
"""
import re
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


# ─────────────────────────────────────────────
# Base Skill
# ─────────────────────────────────────────────
@dataclass
class SkillResult:
    skill_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    annotations: List[str] = field(default_factory=list)  # Human-readable insights
    score: Optional[float] = None                          # 0.0 – 1.0 quality score
    error: Optional[str] = None

    def __str__(self) -> str:
        if not self.success:
            return f"[Skill Error: {self.skill_name}] {self.error}"
        parts = [f"[{self.skill_name}]"]
        if self.score is not None:
            parts.append(f"score={self.score:.2f}")
        parts.extend(self.annotations[:3])
        return " | ".join(parts)


class BaseSkill(ABC):
    """
    Abstract base class for all Skills.
    
    A Skill can be:
    - A post-processor that analyzes/enhances agent output
    - A pre-processor that enriches context before agent runs
    - A validator that checks output quality
    """

    name: str = "base_skill"
    description: str = "A base skill"
    version: str = "1.0"
    tags: List[str] = []

    @abstractmethod
    async def arun(self, context: Dict[str, Any], output: Optional[str] = None) -> SkillResult:
        """
        Execute the skill.
        context: full pipeline context
        output: the agent's text output (for post-processing skills)
        """
        pass

    def run(self, context: Dict[str, Any], output: Optional[str] = None) -> SkillResult:
        return asyncio.get_event_loop().run_until_complete(self.arun(context, output))

    def as_post_hook(self):
        """Return a hook function that can be added to a PipelineStep."""
        async def hook(ctx, agent_output=None):
            result = await self.arun(ctx.all(), agent_output)
            ctx.set(f"skill_{self.name}", result.data)
            return result
        return hook

    def as_pre_hook(self):
        """Return a pre-hook that enriches context before agent runs."""
        async def hook(ctx):
            result = await self.arun(ctx.all())
            ctx.set(f"skill_{self.name}_pre", result.data)
            return result
        return hook

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ─────────────────────────────────────────────
# Skill Registry
# ─────────────────────────────────────────────
class SkillRegistry:
    """Registry for all available skills."""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> "SkillRegistry":
        if not isinstance(skill, BaseSkill):
            raise TypeError(f"Expected BaseSkill, got {type(skill)}")
        self._skills[skill.name] = skill
        return self

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def list(self) -> List[str]:
        return list(self._skills.keys())

    def all(self) -> List[BaseSkill]:
        return list(self._skills.values())

    def by_tag(self, tag: str) -> List[BaseSkill]:
        return [s for s in self._skills.values() if tag in s.tags]

    async def run_all(
        self,
        context: Dict[str, Any],
        output: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, SkillResult]:
        skills = self.by_tag(tags[0]) if tags else self.all()
        tasks  = {s.name: s.arun(context, output) for s in skills}
        results = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as e:
                results[name] = SkillResult(skill_name=name, success=False, error=str(e))
        return results

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={self.list()})"


# ─────────────────────────────────────────────
# Built-in Skills
# ─────────────────────────────────────────────

class WordCountSkill(BaseSkill):
    """Counts words, sentences, paragraphs and sections in text."""
    name = "word_count"
    description = "Analyzes word count, sentence count, paragraphs, and headings"
    tags = ["analysis", "content"]

    async def arun(self, context: Dict, output: Optional[str] = None) -> SkillResult:
        text = output or context.get("editorResult") or context.get("writerResult") or ""
        if not text:
            return SkillResult(skill_name=self.name, success=False, error="No text to analyze")

        words      = len(re.findall(r'\b\w+\b', text))
        sentences  = len(re.findall(r'[.!?]+', text))
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        headings   = len(re.findall(r'^#+\s', text, re.MULTILINE))
        avg_sent   = round(words / max(sentences, 1), 1)

        target  = context.get("wordCount", 1500)
        ratio   = min(words / max(target, 1), 1.0)
        score   = 1.0 - abs(1.0 - ratio) * 0.5  # penalize deviation from target

        annotations = [
            f"共 {words} 词",
            f"{sentences} 句，{paragraphs} 段，{headings} 个标题",
            f"平均句长 {avg_sent} 词",
            f"目标字数达成率 {ratio*100:.0f}%",
        ]

        return SkillResult(
            skill_name=self.name,
            success=True,
            score=round(score, 2),
            data={
                "word_count": words, "sentence_count": sentences,
                "paragraph_count": paragraphs, "heading_count": headings,
                "avg_sentence_length": avg_sent,
                "target_word_count": target, "target_ratio": round(ratio, 2),
            },
            annotations=annotations,
        )


class ReadabilitySkill(BaseSkill):
    """Analyzes text readability using Chinese-adapted metrics."""
    name = "readability"
    description = "Evaluates text readability: sentence variety, vocabulary complexity"
    tags = ["analysis", "quality"]

    async def arun(self, context: Dict, output: Optional[str] = None) -> SkillResult:
        text = output or context.get("editorResult") or context.get("writerResult") or ""
        if not text:
            return SkillResult(skill_name=self.name, success=False, error="No text")

        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if not sentences:
            return SkillResult(skill_name=self.name, success=False, error="Too short")

        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        min_len = min(lengths)
        variety = (max_len - min_len) / max(avg_len, 1)

        # Long sentence penalty
        long_sentences = sum(1 for l in lengths if l > 100)
        long_ratio = long_sentences / len(sentences)

        # Transition words (connectors)
        transitions = len(re.findall(
            r'(因此|所以|但是|然而|此外|另外|首先|其次|最后|总之|综上|不仅|而且|虽然|尽管|例如|比如)',
            text
        ))

        score = min(1.0, max(0.0,
            0.6                           # base
            - long_ratio * 0.4            # penalty for too many long sentences
            + min(transitions / 10, 0.3) # reward connectors
            + min(variety / 5, 0.1)      # reward variety
        ))

        annotations = [
            f"平均句长 {avg_len:.0f} 字符",
            f"长句比例 {long_ratio*100:.0f}%",
            f"过渡词 {transitions} 处",
            f"可读性得分 {score:.0%}",
        ]

        return SkillResult(
            skill_name=self.name, success=True,
            score=round(score, 2),
            data={
                "avg_sentence_length": round(avg_len, 1),
                "sentence_count": len(sentences),
                "long_sentence_ratio": round(long_ratio, 2),
                "transition_word_count": transitions,
                "length_variety": round(variety, 2),
            },
            annotations=annotations,
        )


class SEOSkill(BaseSkill):
    """Analyzes text for SEO quality signals."""
    name = "seo_quality"
    description = "Checks title presence, keyword density, heading structure for SEO"
    tags = ["seo", "quality"]

    async def arun(self, context: Dict, output: Optional[str] = None) -> SkillResult:
        text    = output or context.get("editorResult") or context.get("writerResult") or ""
        seo     = context.get("seoResult") or {}
        outline = context.get("outlineResult") or {}

        if not text:
            return SkillResult(skill_name=self.name, success=False, error="No text")

        checks = {}
        score_parts = []

        # Title check
        has_h1 = bool(re.search(r'^#\s+.+', text, re.MULTILINE))
        checks["has_title"] = has_h1
        score_parts.append(0.2 if has_h1 else 0.0)

        # Heading structure
        h2_count = len(re.findall(r'^##\s+', text, re.MULTILINE))
        has_structure = h2_count >= 2
        checks["heading_count"] = h2_count
        score_parts.append(0.2 if has_structure else 0.1 if h2_count == 1 else 0.0)

        # Keyword usage
        primary_kw = seo.get("primaryKeyword", "")
        kw_count = 0
        if primary_kw and len(primary_kw) > 1:
            kw_count = text.lower().count(primary_kw.lower())
            kw_ok = 1 <= kw_count <= 8
            checks["keyword_in_text"] = kw_ok
            score_parts.append(0.25 if kw_ok else 0.1 if kw_count > 0 else 0.0)
        else:
            score_parts.append(0.15)

        # Meta description present
        has_meta = bool(seo.get("metaDescription"))
        checks["has_meta_description"] = has_meta
        score_parts.append(0.15 if has_meta else 0.0)

        # Content length
        word_count = len(re.findall(r'\b\w+\b', text))
        length_ok = word_count >= 500
        checks["adequate_length"] = length_ok
        score_parts.append(0.2 if word_count >= 1000 else 0.1 if length_ok else 0.0)

        final_score = min(1.0, sum(score_parts))

        annotations = [
            f"标题: {'✓' if has_h1 else '✗'}",
            f"段落标题: {h2_count} 个",
            f"关键词出现 {kw_count} 次",
            f"SEO质量: {final_score:.0%}",
        ]

        return SkillResult(
            skill_name=self.name, success=True,
            score=round(final_score, 2),
            data={**checks, "primary_keyword_count": kw_count, "word_count": word_count},
            annotations=annotations,
        )


class FormatSkill(BaseSkill):
    """Validates and normalizes output format (JSON parsing, Markdown check)."""
    name = "format_validator"
    description = "Validates JSON structure and Markdown formatting"
    tags = ["format", "validation"]

    def __init__(self, expected_format: str = "markdown"):
        self.expected_format = expected_format  # "markdown" | "json"

    async def arun(self, context: Dict, output: Optional[str] = None) -> SkillResult:
        text = output or ""
        if not text:
            return SkillResult(skill_name=self.name, success=False, error="Empty output")

        if self.expected_format == "json":
            # Try to parse JSON
            clean = re.sub(r'^```json\s*', '', text.strip())
            clean = re.sub(r'```\s*$', '', clean.strip())
            try:
                parsed = json.loads(clean)
                return SkillResult(
                    skill_name=self.name, success=True, score=1.0,
                    data={"valid_json": True, "parsed": parsed},
                    annotations=["JSON 格式有效"],
                )
            except json.JSONDecodeError as e:
                return SkillResult(
                    skill_name=self.name, success=False, score=0.0,
                    error=f"JSON 解析失败: {e}",
                    data={"valid_json": False},
                )
        else:
            # Markdown checks
            has_h1   = bool(re.search(r'^#\s', text, re.MULTILINE))
            has_h2   = bool(re.search(r'^##\s', text, re.MULTILINE))
            has_para = len([p for p in text.split('\n\n') if p.strip()]) >= 2
            score    = (0.4 * has_h1 + 0.3 * has_h2 + 0.3 * has_para)

            return SkillResult(
                skill_name=self.name, success=True,
                score=round(score, 2),
                data={"has_h1": has_h1, "has_h2": has_h2, "has_paragraphs": has_para},
                annotations=[
                    f"H1标题: {'✓' if has_h1 else '✗'}",
                    f"H2章节: {'✓' if has_h2 else '✗'}",
                    f"多段落: {'✓' if has_para else '✗'}",
                ],
            )


class ToneAnalysisSkill(BaseSkill):
    """Analyzes writing tone against target style."""
    name = "tone_analysis"
    description = "Checks if text tone matches the requested writing style"
    tags = ["analysis", "quality"]

    STYLE_SIGNALS = {
        "专业干货": {
            "positive": ["数据", "研究", "分析", "结论", "因此", "通过", "方法", "策略"],
            "negative": ["哈哈", "吧", "呢", "嗯", "感觉"],
        },
        "批判性分析": {
            "positive": ["然而", "但是", "值得注意", "问题在于", "质疑", "反思", "局限"],
            "negative": ["完美", "最好", "无可挑剔"],
        },
        "通俗易懂": {
            "positive": ["比如", "就是说", "简单来说", "想象一下", "举个例子"],
            "negative": ["此外", "综上所述", "鉴于", "基于"],
        },
        "故事叙述": {
            "positive": ["突然", "那天", "记得", "当时", "后来", "最终", "他说", "她"],
            "negative": ["数据表明", "研究显示", "据统计"],
        },
    }

    async def arun(self, context: Dict, output: Optional[str] = None) -> SkillResult:
        text  = output or context.get("editorResult") or ""
        style = context.get("style", "专业干货")

        signals = self.STYLE_SIGNALS.get(style, self.STYLE_SIGNALS["专业干货"])
        pos_hits = sum(text.count(w) for w in signals["positive"])
        neg_hits = sum(text.count(w) for w in signals["negative"])

        score = min(1.0, max(0.0, (pos_hits * 0.1 - neg_hits * 0.05 + 0.5)))

        return SkillResult(
            skill_name=self.name, success=True,
            score=round(score, 2),
            data={
                "target_style": style,
                "positive_signals": pos_hits,
                "negative_signals": neg_hits,
            },
            annotations=[
                f"目标风格: {style}",
                f"风格匹配信号: {pos_hits} 正 / {neg_hits} 负",
                f"风格一致性: {score:.0%}",
            ],
        )


# ─────────────────────────────────────────────
# Default skill set factory
# ─────────────────────────────────────────────
def create_default_skills() -> SkillRegistry:
    """Create and return the default skill registry."""
    registry = SkillRegistry()
    registry.register(WordCountSkill())
    registry.register(ReadabilitySkill())
    registry.register(SEOSkill())
    registry.register(FormatSkill(expected_format="markdown"))
    registry.register(ToneAnalysisSkill())
    return registry
