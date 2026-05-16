"""
ContentForge — Six Specialized Sub-Agents
══════════════════════════════════════════
每个 Agent 继承自 langchain_core.BaseAgent，
实现 get_system_prompt() 和 build_messages()。
输出通过 post_process() 进行结构化处理。
"""
import json
import re
from typing import Any, Dict, List, Optional

from ..langchain_core.agents import BaseAgent
from ..langchain_core.messages import BaseMessage, HumanMessage


# ─── JSON 安全解析 ─────────────────────────────
def safe_json(text: str, fallback: Any = None) -> Any:
    """Strip markdown fences and parse JSON safely."""
    clean = re.sub(r'^```json\s*', '', text.strip())
    clean = re.sub(r'^```\s*', '', clean)
    clean = re.sub(r'```\s*$', '', clean.strip())
    try:
        return json.loads(clean)
    except Exception:
        # Try to extract JSON object from text
        match = re.search(r'\{[\s\S]+\}', clean)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return fallback


# ═══════════════════════════════════════════════
# 1. Topic Agent — 选题策划师
# ═══════════════════════════════════════════════
class TopicAgent(BaseAgent):
    id          = "topic"
    name        = "选题策划师"
    description = "分析需求，策划最优写作角度与核心钩子"
    icon        = "🎯"
    color       = "#8b5cf6"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=800, **kwargs)
        self.llm.max_tokens = 800

    def get_system_prompt(self) -> str:
        return """你是一位资深内容策划师，擅长从用户的粗糙需求中提炼出最具传播力的写作角度。

你的任务：
1. 分析目标受众与传播平台
2. 提出 3 个不同角度的选题方向（差异化，有对比性）
3. 推荐最优选题并说明选择理由
4. 确定核心 Hook（读者痛点或好奇心锚点）

输出格式（严格 JSON，不加任何其他文字）：
{
  "topic": "最终选定的主题",
  "angle": "具体写作角度",
  "hook": "开篇钩子句（一句话抓住读者）",
  "audience": "目标受众描述",
  "reason": "选择该角度的理由",
  "alternatives": ["备选方向1", "备选方向2"]
}"""

    async def build_messages(self, context: Dict[str, Any]) -> List[BaseMessage]:
        return [HumanMessage(content=
            f"用户需求：{context.get('userRequest', '')}\n"
            f"目标平台：{context.get('platform', '通用媒体')}\n"
            f"写作风格：{context.get('style', '专业干货')}\n\n"
            "请策划最佳选题方向，输出 JSON。"
        )]

    def post_process(self, raw: str, context: Dict) -> Dict:
        result = safe_json(raw, fallback={})
        if not result.get("topic"):
            result["topic"] = context.get("userRequest", "")
        return result


# ═══════════════════════════════════════════════
# 2. Research Agent — 调研分析师
# ═══════════════════════════════════════════════
class ResearchAgent(BaseAgent):
    id          = "research"
    name        = "调研分析师"
    description = "收集背景知识、关键数据与典型案例"
    icon        = "🔬"
    color       = "#06b6d4"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=1200, **kwargs)
        self.llm.max_tokens = 1200

    def get_system_prompt(self) -> str:
        return """你是一位专业的内容调研分析师。基于给定的写作主题，你需要：

1. 梳理该话题的背景知识框架（2-3段）
2. 列举 4-6 个有说服力的数据点或事实（可使用参考数据，标注"[参考数据]"）
3. 找出 2-3 个典型案例或生动类比
4. 识别 2-3 个常见误区（为写作提供对比素材）
5. 定义 3-5 个核心概念

输出格式（严格 JSON）：
{
  "background": "背景知识摘要文字",
  "keyFacts": ["数据/事实1（来源/标注）", "事实2"],
  "cases": ["案例1：具体描述", "案例2"],
  "misconceptions": ["误区1：XXX其实并非YYY", "误区2"],
  "definitions": {"概念名": "定义说明"}
}"""

    async def build_messages(self, context: Dict) -> List[BaseMessage]:
        topic = context.get("topicResult") or {}
        return [HumanMessage(content=
            f"写作主题：{topic.get('topic', context.get('userRequest', ''))}\n"
            f"写作角度：{topic.get('angle', '')}\n"
            f"目标受众：{topic.get('audience', '')}\n\n"
            "请进行深度调研，提供高质量素材，输出 JSON。"
        )]

    def post_process(self, raw: str, context: Dict) -> Dict:
        result = safe_json(raw, fallback={})
        defaults = {"background": "", "keyFacts": [], "cases": [], "misconceptions": [], "definitions": {}}
        return {**defaults, **result}


# ═══════════════════════════════════════════════
# 3. Outline Agent — 结构设计师
# ═══════════════════════════════════════════════
class OutlineAgent(BaseAgent):
    id          = "outline"
    name        = "结构设计师"
    description = "设计文章骨架与章节逻辑"
    icon        = "🗂️"
    color       = "#f59e0b"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=900, **kwargs)
        self.llm.max_tokens = 900

    def get_system_prompt(self) -> str:
        return """你是专业的内容结构设计师，擅长构建逻辑严密、引人入胜的文章骨架。

你的任务：
1. 设计符合读者认知规律的章节顺序（问题→分析→解决方案→结论）
2. 为每个章节确定核心论点（一句话）
3. 指定每节需要引用的调研素材
4. 预估字数分配（总计符合目标）
5. 设计强有力的标题（含关键词，有吸引力）

输出格式（严格 JSON）：
{
  "title": "文章主标题（吸引眼球，含核心关键词）",
  "subtitle": "副标题（补充说明，可选）",
  "intro": "引言写法提示（1-2句）",
  "sections": [
    {
      "heading": "章节小标题",
      "point": "该节核心论点（一句话）",
      "materials": ["引用的素材关键词"],
      "wordCount": 300
    }
  ],
  "conclusion": "结尾思路（1-2句）",
  "totalWords": 1500
}"""

    async def build_messages(self, context: Dict) -> List[BaseMessage]:
        topic    = context.get("topicResult")    or {}
        research = context.get("researchResult") or {}
        return [HumanMessage(content=
            f"选题信息：\n"
            f"  主题：{topic.get('topic', '')}\n"
            f"  角度：{topic.get('angle', '')}\n"
            f"  受众：{topic.get('audience', '')}\n"
            f"  钩子：{topic.get('hook', '')}\n\n"
            f"调研素材：\n"
            f"  背景：{research.get('background', '')[:300]}\n"
            f"  数据：{'; '.join(research.get('keyFacts', [])[:3])}\n"
            f"  案例：{'; '.join(research.get('cases', [])[:2])}\n\n"
            f"目标字数：{context.get('wordCount', 1500)} 字\n"
            f"写作风格：{context.get('style', '专业干货')}\n\n"
            "请设计文章大纲，输出 JSON。"
        )]

    def post_process(self, raw: str, context: Dict) -> Dict:
        result = safe_json(raw, fallback={})
        if not result.get("sections"):
            result["sections"] = []
        if not result.get("totalWords"):
            result["totalWords"] = context.get("wordCount", 1500)
        return result


# ═══════════════════════════════════════════════
# 4. Writer Agent — 内容撰写师
# ═══════════════════════════════════════════════
class WriterAgent(BaseAgent):
    id          = "writer"
    name        = "内容撰写师"
    description = "依据大纲和素材生成完整正文"
    icon        = "✍️"
    color       = "#10b981"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=2800, **kwargs)
        self.llm.max_tokens = 2800

    def get_system_prompt(self) -> str:
        return """你是一位顶级内容创作者，文笔流畅，观点深刻，擅长将复杂知识以通俗易懂的语言呈现。

写作铁律：
1. 【结构遵从】严格按照大纲章节展开，不缺段，不跑题
2. 【素材充分】充分引用调研数据、案例、定义，增强说服力
3. 【语言鲜活】用具体数字、生动比喻替代空洞描述
4. 【逻辑清晰】每段一个核心意思，段段有论点有论据
5. 【过渡流畅】章节之间有自然的承上启下
6. 【标题格式】章节标题用 ## 格式，有需要用 ### 三级标题
7. 【钩子开篇】引言必须用提供的钩子句切入，瞬间抓住读者

直接输出 Markdown 格式完整文章，从 # 标题开始，不加任何说明性文字。"""

    async def build_messages(self, context: Dict) -> List[BaseMessage]:
        topic    = context.get("topicResult")    or {}
        research = context.get("researchResult") or {}
        outline  = context.get("outlineResult")  or {}

        sections_text = "\n".join([
            f"  {i+1}. {s.get('heading','')}（{s.get('wordCount',300)}字）\n"
            f"     论点：{s.get('point','')}"
            for i, s in enumerate(outline.get("sections", []))
        ])

        facts_text  = "\n".join(f"  - {f}" for f in research.get("keyFacts", []))
        cases_text  = "\n".join(f"  - {c}" for c in research.get("cases", []))
        miscon_text = "\n".join(f"  - {m}" for m in research.get("misconceptions", []))

        return [HumanMessage(content=
            f"## 写作任务\n\n"
            f"**标题**：{outline.get('title', topic.get('topic', ''))}\n"
            f"**开篇钩子**：{topic.get('hook', '')}\n"
            f"**目标受众**：{topic.get('audience', '')}\n"
            f"**写作风格**：{context.get('style', '专业干货')}\n"
            f"**目标字数**：约 {outline.get('totalWords', context.get('wordCount', 1500))} 字\n\n"
            f"## 章节大纲\n{sections_text}\n\n"
            f"## 可用素材\n\n"
            f"**关键数据**：\n{facts_text}\n\n"
            f"**典型案例**：\n{cases_text}\n\n"
            f"**常见误区**（可作为反驳论据）：\n{miscon_text}\n\n"
            f"**背景知识**：\n{research.get('background', '')[:400]}\n\n"
            "请输出完整的 Markdown 文章。"
        )]


# ═══════════════════════════════════════════════
# 5. Editor Agent — 资深编辑
# ═══════════════════════════════════════════════
class EditorAgent(BaseAgent):
    id          = "editor"
    name        = "资深编辑"
    description = "润色文章语言，提升表达质量"
    icon        = "✏️"
    color       = "#ef4444"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=3000, **kwargs)
        self.llm.max_tokens = 3000

    def get_system_prompt(self) -> str:
        return """你是一位资深文字编辑，拥有极高的文字审美和逻辑敏感度。

编辑优先级（从高到低）：
1. **语言问题**：修正所有语病、表达不当、词语误用、标点错误
2. **逻辑问题**：确保每个论点都有充分支撑，删除空洞断言
3. **流畅性**：优化段落过渡句，让文章如流水般顺畅
4. **节奏感**：调整句子长短搭配，避免连续长句或短句
5. **开结强化**：开篇要一句话抓人，结尾要有力量和回味
6. **保留骨架**：不改变文章结构、标题体系和核心观点

绝对禁止：不添加新观点、不改变结论、不改动标题格式。

直接输出润色后的完整 Markdown 文章，不加任何说明或注释。"""

    async def build_messages(self, context: Dict) -> List[BaseMessage]:
        topic = context.get("topicResult") or {}
        draft = context.get("writerResult") or ""
        return [HumanMessage(content=
            f"请润色以下文章：\n\n"
            f"写作风格要求：{context.get('style', '专业干货')}\n"
            f"目标受众：{topic.get('audience', '普通读者')}\n\n"
            f"---\n\n{draft}"
        )]


# ═══════════════════════════════════════════════
# 6. SEO Agent — SEO 优化师
# ═══════════════════════════════════════════════
class SEOAgent(BaseAgent):
    id          = "seo"
    name        = "SEO 优化师"
    description = "提取关键词，生成 Meta 信息，评估 SEO 质量"
    icon        = "📈"
    color       = "#f97316"

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, max_tokens=800, **kwargs)
        self.llm.max_tokens = 800

    def get_system_prompt(self) -> str:
        return """你是专业的 SEO 内容优化师，精通搜索引擎优化策略。

根据文章内容，输出完整的 SEO 分析报告：
1. 主关键词（1个，搜索量最大、最符合用户意图）
2. 长尾关键词（5个，自然融入文章的变体）
3. Meta Title（60字符以内，含主关键词，有点击欲望）
4. Meta Description（155字符以内，包含CTA，吸引用户点击）
5. Open Graph 标题（社交媒体分享标题，可比Meta Title更口语化）
6. 文章标签 Tags（5个，用于分类和发现）
7. SEO 质量评分（0-100整数）
8. 优化建议（2-3条具体可执行的建议）

输出格式（严格 JSON）：
{
  "primaryKeyword": "主关键词",
  "longtailKeywords": ["词1", "词2", "词3", "词4", "词5"],
  "metaTitle": "...",
  "metaDescription": "...",
  "ogTitle": "...",
  "tags": ["标签1", "标签2", "标签3", "标签4", "标签5"],
  "seoScore": 85,
  "suggestions": ["建议1", "建议2", "建议3"]
}"""

    async def build_messages(self, context: Dict) -> List[BaseMessage]:
        topic   = context.get("topicResult")  or {}
        article = context.get("editorResult") or context.get("writerResult") or ""
        return [HumanMessage(content=
            f"请对以下文章进行 SEO 分析：\n\n"
            f"主题：{topic.get('topic', context.get('userRequest', ''))}\n"
            f"目标受众：{topic.get('audience', '')}\n\n"
            f"文章内容（前1200字）：\n{article[:1200]}...\n\n"
            "输出 JSON。"
        )]

    def post_process(self, raw: str, context: Dict) -> Dict:
        result = safe_json(raw, fallback={})
        defaults = {
            "primaryKeyword": "", "longtailKeywords": [],
            "metaTitle": "", "metaDescription": "", "ogTitle": "",
            "tags": [], "seoScore": 0, "suggestions": [],
        }
        return {**defaults, **result}


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────
def create_all_agents(llm) -> List[BaseAgent]:
    """Create all six content agents sharing the same LLM."""
    import copy
    return [
        TopicAgent(llm=copy.copy(llm)),
        ResearchAgent(llm=copy.copy(llm)),
        OutlineAgent(llm=copy.copy(llm)),
        WriterAgent(llm=copy.copy(llm)),
        EditorAgent(llm=copy.copy(llm)),
        SEOAgent(llm=copy.copy(llm)),
    ]
