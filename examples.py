#!/usr/bin/env python3
"""
ContentForge — 使用示例与单元测试
════════════════════════════════════════════

示例 1：基本使用
示例 2：挂载自定义 Skill
示例 3：自定义 Agent
示例 4：单元测试（无需 API）
"""
import sys
import asyncio
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════
# 示例 1 — 基本使用
# ═══════════════════════════════════════════════
def example_basic():
    """最简单的使用方式。"""
    from contentforge_py.master import create_master, ContentRequest

    # ┌─────────────────────────────────────────┐
    # │  填入你的 API Key                        │
    # └─────────────────────────────────────────┘
    master = create_master(
        provider = "anthropic",           # anthropic | openai | custom
        api_key  = "YOUR_API_KEY_HERE",   # ← 填入 API Key
        model    = "",                    # 留空使用默认模型
        verbose  = True,
    )

    result = master.create(ContentRequest(
        user_request = "写一篇关于 AI 大模型工程化落地挑战的深度分析，面向技术管理者",
        platform     = "知乎",
        style        = "专业干货",
        word_count   = 2000,
    ))

    if result.success:
        print(f"\n✅ 文章标题: {result.outline.get('title', '')}")
        print(f"   SEO 评分: {result.seo.get('seoScore', 0)}/100")
        print(f"   主关键词: {result.seo.get('primaryKeyword', '')}")
        print(f"   耗时: {result.duration:.1f}s\n")

        # Save
        Path("output_basic.md").write_text(result.article, encoding="utf-8")
        print("💾 文章已保存: output_basic.md")
    else:
        print(f"❌ 失败: {result.error}")


# ═══════════════════════════════════════════════
# 示例 2 — 挂载自定义 Skill
# ═══════════════════════════════════════════════
def example_custom_skill():
    """
    展示如何编写并挂载自定义 Skill。
    本例：KeywordDensitySkill — 检查关键词密度是否合理。
    """
    import re
    from contentforge_py.harness.skills import BaseSkill, SkillResult
    from contentforge_py.master import create_master, ContentRequest

    class KeywordDensitySkill(BaseSkill):
        """检查主关键词的密度是否在 1%-3% 之间（SEO 最佳实践）。"""
        name        = "keyword_density"
        description = "Checks primary keyword density (ideal: 1%-3%)"
        tags        = ["seo", "custom"]

        async def arun(self, context, output=None):
            text = output or context.get("editorResult") or ""
            seo  = context.get("seoResult") or {}
            kw   = seo.get("primaryKeyword", "")

            if not text or not kw:
                return SkillResult(
                    skill_name=self.name, success=False,
                    error="缺少文章或关键词"
                )

            words     = re.findall(r'\w+', text)
            kw_count  = text.lower().count(kw.lower())
            density   = kw_count / max(len(words), 1) * 100
            in_range  = 1.0 <= density <= 3.0
            score     = 1.0 if in_range else max(0.0, 1.0 - abs(density - 2.0) * 0.3)

            return SkillResult(
                skill_name=self.name,
                success=True,
                score=round(score, 2),
                data={"keyword": kw, "count": kw_count, "density_pct": round(density, 2)},
                annotations=[
                    f"关键词「{kw}」出现 {kw_count} 次",
                    f"密度 {density:.2f}%（理想 1%-3%）",
                    f"{'✓ 达标' if in_range else '⚠️ 需调整'}",
                ],
            )

    master = create_master(
        provider = "anthropic",
        api_key  = "YOUR_API_KEY_HERE",
        skills   = [KeywordDensitySkill()],  # 挂载自定义 Skill
        verbose  = True,
    )

    result = master.create(ContentRequest(
        user_request = "写一篇 Python 异步编程入门教程",
        style        = "通俗易懂",
        word_count   = 1200,
    ))

    if result.success:
        kd = result.skills.get("keyword_density", {})
        print(f"\n💡 Keyword Density Skill 结果:")
        for ann in (kd.get("annotations") or []):
            print(f"   {ann}")


# ═══════════════════════════════════════════════
# 示例 3 — 自定义 Agent（添加"翻译 Agent"）
# ═══════════════════════════════════════════════
def example_custom_agent():
    """
    展示如何添加自定义 Agent 到 Pipeline。
    本例：TranslatorAgent — 将文章翻译成英文。
    """
    from contentforge_py.langchain_core.agents import BaseAgent
    from contentforge_py.langchain_core.messages import HumanMessage
    from contentforge_py.harness.engine import PipelineStep
    from contentforge_py.master import create_master, ContentRequest, build_pipeline
    from contentforge_py.langchain_core.llm import create_llm
    import copy

    class TranslatorAgent(BaseAgent):
        id          = "translator"
        name        = "翻译专家"
        description = "将中文文章翻译为流畅的英文"
        icon        = "🌐"
        color       = "#3b82f6"

        def get_system_prompt(self) -> str:
            return ("你是专业的中英翻译，请将给定的中文文章翻译成流畅自然的英文。"
                    "保持原文的Markdown格式，标题也翻译。只输出翻译结果。")

        async def build_messages(self, context):
            article = context.get("editorResult") or ""
            return [HumanMessage(content=f"请翻译以下文章：\n\n{article}")]

    # Build master
    llm = create_llm(provider="anthropic", api_key="YOUR_API_KEY_HERE")

    from contentforge_py.master import MasterAgent
    from contentforge_py.harness.skills import create_default_skills

    master = MasterAgent(
        llm=llm,
        skill_registry=create_default_skills(),
        verbose=True,
    )

    # Register the custom agent
    translator = TranslatorAgent(llm=copy.copy(llm))
    master.harness.register(translator)

    # Extend the pipeline with the translation step
    from contentforge_py.master import build_pipeline as bp
    pipeline = bp() + [
        PipelineStep(
            agent_id   = "translator",
            output_key = "translationResult",
            name       = "英文翻译",
            description= "将最终文章翻译为英文",
            retries    = 1,
            timeout    = 90.0,
        )
    ]

    # Manually run
    loop = asyncio.new_event_loop()
    result_data = loop.run_until_complete(master.harness.run(
        steps=pipeline,
        initial_context={
            "userRequest": "写一篇关于量子计算的科普文章",
            "platform": "通用媒体",
            "style": "通俗易懂",
            "wordCount": 1000,
        }
    ))
    loop.close()

    translation = master.harness.context.get("translationResult") or ""
    print(f"\n🌐 英文翻译前 200 字符:\n{translation[:200]}...")


# ═══════════════════════════════════════════════
# 示例 4 — 单元测试（无需 API Key）
# ═══════════════════════════════════════════════
def run_unit_tests():
    """Tests that don't require real API calls."""
    import traceback

    passed = 0
    failed = 0

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n🧪 ContentForge 单元测试\n")

    # ── langchain_core tests ────────────────
    def test_messages():
        from contentforge_py.langchain_core.messages import HumanMessage, AIMessage, SystemMessage, messages_to_dicts
        h = HumanMessage("Hello")
        a = AIMessage("World")
        s = SystemMessage("System")
        assert h.role.value == "user"
        assert a.role.value == "assistant"
        assert s.role.value == "system"
        dicts = messages_to_dicts([h, a])
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"

    def test_prompt_template():
        from contentforge_py.langchain_core.prompts import PromptTemplate, ChatPromptTemplate
        pt = PromptTemplate.from_template("Hello {name}, you are {age} years old.")
        assert "name" in pt.input_variables
        assert "age" in pt.input_variables
        result = pt.format(name="Alice", age=30)
        assert "Alice" in result and "30" in result

        cpt = ChatPromptTemplate.from_messages([
            ("system", "You are {role}"),
            ("human", "Question: {question}"),
        ])
        msgs = cpt.format_messages(role="helper", question="What is AI?")
        assert len(msgs) == 2

    def test_context_bus():
        from contentforge_py.harness.engine import ContextBus
        ctx = ContextBus({"a": 1})
        ctx.set("b", 2)
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2
        assert ctx.get("missing", "default") == "default"
        assert "a" in ctx
        snap = ctx.snapshot()
        assert snap == {"a": 1, "b": 2}
        ctx.reset()
        assert ctx.get("a") is None

    def test_tool_registry():
        from contentforge_py.langchain_core.tools import BaseTool, ToolRegistry
        class DummyTool(BaseTool):
            name = "dummy"
            description = "A dummy tool"
            async def _arun(self, **kwargs): return "done"

        reg = ToolRegistry()
        reg.register(DummyTool())
        assert "dummy" in reg
        assert len(reg) == 1
        assert reg.get("dummy") is not None
        assert reg.get("missing") is None

    def test_skill_registry():
        from contentforge_py.harness.skills import SkillRegistry, WordCountSkill, ReadabilitySkill
        reg = SkillRegistry()
        reg.register(WordCountSkill())
        reg.register(ReadabilitySkill())
        assert "word_count" in reg
        assert "readability" in reg
        assert len(reg) == 2

    def test_word_count_skill():
        from contentforge_py.harness.skills import WordCountSkill
        skill = WordCountSkill()
        text  = "这是一个测试文章。" * 100
        ctx   = {"wordCount": 500}
        loop  = asyncio.new_event_loop()
        result = loop.run_until_complete(skill.arun(ctx, text))
        loop.close()
        assert result.success
        assert result.score is not None
        assert "word_count" in result.data

    def test_readability_skill():
        from contentforge_py.harness.skills import ReadabilitySkill
        skill = ReadabilitySkill()
        text  = "人工智能正在快速发展。因此，很多行业都受到影响。但是，我们不必担心。"
        loop  = asyncio.new_event_loop()
        result = loop.run_until_complete(skill.arun({}, text))
        loop.close()
        assert result.success
        assert 0.0 <= result.score <= 1.0

    def test_format_skill_markdown():
        from contentforge_py.harness.skills import FormatSkill
        skill = FormatSkill(expected_format="markdown")
        text  = "# 标题\n\n## 章节一\n\n这是内容。\n\n## 章节二\n\n更多内容。"
        loop  = asyncio.new_event_loop()
        result = loop.run_until_complete(skill.arun({}, text))
        loop.close()
        assert result.success
        assert result.score >= 0.7

    def test_format_skill_json():
        from contentforge_py.harness.skills import FormatSkill
        skill = FormatSkill(expected_format="json")
        # Valid JSON
        loop = asyncio.new_event_loop()
        r1 = loop.run_until_complete(skill.arun({}, '{"key": "value"}'))
        assert r1.success and r1.score == 1.0
        # Invalid JSON
        r2 = loop.run_until_complete(skill.arun({}, 'not json'))
        assert not r2.success
        loop.close()

    def test_safe_json():
        from contentforge_py.agents.content_agents import safe_json
        assert safe_json('{"a":1}') == {"a": 1}
        assert safe_json('```json\n{"b":2}\n```') == {"b": 2}
        assert safe_json("not json", fallback={"x": 0}) == {"x": 0}
        # Embedded JSON
        result = safe_json('Some text {"c": 3} more text')
        assert result.get("c") == 3

    def test_harness_registration():
        from contentforge_py.harness.engine import Harness
        from contentforge_py.langchain_core.agents import BaseAgent
        from contentforge_py.langchain_core.messages import HumanMessage

        class DummyAgent(BaseAgent):
            id = "dummy_h"
            name = "Dummy"
            def get_system_prompt(self): return "You are dummy."
            async def build_messages(self, context): return [HumanMessage("test")]

        h = Harness(verbose=False)
        h.register(DummyAgent())
        assert "dummy_h" in h.registry
        assert len(h.registry) == 1

    def test_pipeline_step():
        from contentforge_py.harness.engine import PipelineStep, ContextBus
        ctx = ContextBus({"ready": True})
        step1 = PipelineStep(agent_id="x", output_key="xOut")
        assert step1.should_run(ctx) is True

        step2 = PipelineStep(
            agent_id="y", output_key="yOut",
            condition=lambda c: c.get("ready") is True
        )
        assert step2.should_run(ctx) is True

        step3 = PipelineStep(
            agent_id="z", output_key="zOut",
            condition=lambda c: c.get("missing") is True
        )
        assert step3.should_run(ctx) is False

    def test_event_bus():
        from contentforge_py.langchain_core.callbacks import EventBus, EventType
        bus = EventBus()
        received = []
        bus.on("*", lambda e: received.append(e.type))
        bus.emit(EventType.LOG, message="test")
        bus.emit(EventType.STEP_START, step_id="x")
        assert EventType.LOG in received
        assert EventType.STEP_START in received

    # Run all tests
    test("Messages",            test_messages)
    test("PromptTemplate",      test_prompt_template)
    test("ContextBus",          test_context_bus)
    test("ToolRegistry",        test_tool_registry)
    test("SkillRegistry",       test_skill_registry)
    test("WordCountSkill",      test_word_count_skill)
    test("ReadabilitySkill",    test_readability_skill)
    test("FormatSkill(md)",     test_format_skill_markdown)
    test("FormatSkill(json)",   test_format_skill_json)
    test("safe_json",           test_safe_json)
    test("Harness.register",    test_harness_registration)
    test("PipelineStep.cond",   test_pipeline_step)
    test("EventBus",            test_event_bus)

    total = passed + failed
    print(f"\n{'─'*40}")
    print(f"结果: {passed}/{total} 通过", end="")
    if failed:
        print(f"，{failed} 失败 ⚠️")
    else:
        print(" ✅")
    return failed == 0


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        ok = run_unit_tests()
        sys.exit(0 if ok else 1)
    elif mode == "basic":
        example_basic()
    elif mode == "skill":
        example_custom_skill()
    elif mode == "agent":
        example_custom_agent()
    else:
        print("用法: python examples.py [test|basic|skill|agent]")
