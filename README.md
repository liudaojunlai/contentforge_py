# ContentForge Python
## 智能内容创作 Multi-Agent 系统

> 基于自研 LangChain-style 框架 + Harness 工程模式，纯 Python 标准库实现，零外部依赖。
> 支持 Anthropic / OpenAI / 自定义 API，可插拔 Skill 系统。

---

## 🏗️ 架构总览

```
用户请求
    │
    ▼
┌──────────────────────────────────────────────┐
│               MasterAgent                    │
│  ┌─────────────────────────────────────────┐ │
│  │         Harness Engine                  │ │
│  │  ContextBus · AgentRegistry · EventBus  │ │
│  └──────────────┬──────────────────────────┘ │
└─────────────────┼────────────────────────────┘
                  │ Pipeline (串行 + 条件 + 并行)
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 TopicAgent  ResearchAgent  OutlineAgent
    │             │             │
    └─────────────┼─────────────┘
                  ▼
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
 WriterAgent  EditorAgent   SEOAgent
                  │
                  │  post_hooks (Skills)
    ┌─────────────┼──────────────────────┐
    ▼             ▼                      ▼
WordCount  Readability  SEOQuality  ToneAnalysis  FormatValidator
```

---

## 📁 项目结构

```
contentforge_py/
├── __init__.py
├── master.py                  # MasterAgent — 顶层调度器
├── cli.py                     # 命令行界面
├── server.py                  # HTTP API 服务器（stdlib only）
├── examples.py                # 使用示例 & 单元测试
│
├── langchain_core/            # 自研 LangChain-style 核心框架
│   ├── messages.py            # HumanMessage / AIMessage / SystemMessage
│   ├── prompts.py             # PromptTemplate / ChatPromptTemplate
│   ├── llm.py                 # BaseLLM / ChatAnthropic / ChatOpenAI / ChatCustom
│   ├── chains.py              # LLMChain / SequentialChain / TransformChain
│   ├── memory.py              # ConversationBufferMemory / SummaryMemory
│   ├── tools.py               # BaseTool / ToolResult / ToolRegistry
│   ├── agents.py              # BaseAgent / AgentExecutor
│   ├── callbacks.py           # EventBus / CallbackManager / ConsoleHandler
│   └── schema.py              # BaseRunnable / RunnableConfig
│
├── harness/                   # Harness 调度引擎
│   ├── engine.py              # Harness / PipelineStep / ContextBus / AgentRegistry
│   └── skills.py              # BaseSkill / SkillRegistry / 5个内置Skill
│
├── agents/                    # 六个专业子 Agent
│   └── content_agents.py      # TopicAgent → SEOAgent
│
└── ui/
    └── index.html             # Web UI（支持直连 & 服务器两种模式）
```

---

## 🤖 六个子 Agent

| Agent | 职责 | 输出类型 |
|-------|------|---------|
| 🎯 TopicAgent | 分析需求，策划角度与钩子 | JSON |
| 🔬 ResearchAgent | 收集背景、数据、案例、误区 | JSON |
| 🗂️ OutlineAgent | 设计章节结构与字数分配 | JSON |
| ✍️ WriterAgent | 依据大纲生成完整正文 | Markdown |
| ✏️ EditorAgent | 润色语言，修正逻辑 | Markdown |
| 📈 SEOAgent | 关键词分析，生成 Meta 信息 | JSON |

---

## 💡 五个内置 Skill

| Skill | 职责 | 触发时机 |
|-------|------|---------|
| `word_count` | 字数、句数、段落统计 | EditorAgent 完成后 |
| `readability` | 可读性评分（句长、过渡词） | EditorAgent 完成后 |
| `seo_quality` | SEO 质量检查 | EditorAgent 完成后 |
| `tone_analysis` | 语气与风格一致性检测 | EditorAgent 完成后 |
| `format_validator` | Markdown 格式校验 | WriterAgent 完成后 |

---

## ⚙️ Harness 工程模式

```python
from contentforge_py.harness.engine import Harness, PipelineStep

harness = Harness(name="my-pipeline")
harness.register(my_agent)

steps = [
    PipelineStep(
        agent_id    = "my_agent",
        output_key  = "result",
        retries     = 2,
        timeout     = 60.0,
        condition   = lambda ctx: ctx.get("ready") is True,  # 条件跳过
        pre_hooks   = [my_pre_fn],    # Agent 执行前
        post_hooks  = [skill.as_post_hook()],  # Agent 完成后
    )
]

await harness.run(steps, initial_context={"ready": True})
```

---

## 🚀 快速开始

### 方式一：Web UI（推荐）

直接用浏览器打开 `ui/index.html`，填入 API Key 即可使用。

或启动 Python 服务器（API Key 由服务器管理）：

```bash
# 设置环境变量
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-your-key-here

# 启动服务器
python -m contentforge_py.server --port 8080

# 访问 http://localhost:8080
```

### 方式二：命令行

```bash
# 单次创作
python -m contentforge_py.cli \
  --provider anthropic \
  --key sk-ant-your-key \
  --request "写一篇关于AI大模型工程化落地的深度分析" \
  --platform 知乎 \
  --style 批判性分析 \
  --words 2000

# 交互模式
python -m contentforge_py.cli \
  --provider anthropic \
  --key sk-ant-your-key \
  --interactive

# 使用环境变量
export LLM_API_KEY=sk-ant-your-key
python -m contentforge_py.cli --request "你的需求"
```

### 方式三：Python API

```python
from contentforge_py.master import create_master, ContentRequest

# ════════════════════════════════════
# 填入 API Key
# ════════════════════════════════════
master = create_master(
    provider = "anthropic",         # anthropic | openai | custom
    api_key  = "YOUR_KEY_HERE",     # ← 填入这里
    model    = "",                  # 留空使用默认
)

result = master.create(ContentRequest(
    user_request = "写一篇关于AI的深度分析",
    platform     = "知乎",
    style        = "专业干货",
    word_count   = 2000,
))

print(result.article)       # Markdown 文章
print(result.seo)           # SEO 数据
print(result.skills)        # Skill 分析结果
```

---

## 🔑 API Key 配置方式

**方式 1：代码中（master.py `create_master` 函数）**
```python
master = create_master(provider="anthropic", api_key="sk-ant-xxx")
```

**方式 2：环境变量**
```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-xxx
export LLM_MODEL=           # 可选
export LLM_BASE_URL=        # custom provider 专用
```

**方式 3：CLI 参数**
```bash
python -m contentforge_py.cli --provider anthropic --key sk-ant-xxx
```

---

## 🔧 扩展：自定义 Skill

```python
from contentforge_py.harness.skills import BaseSkill, SkillResult

class MySkill(BaseSkill):
    name        = "my_skill"
    description = "My custom analysis"
    tags        = ["custom"]

    async def arun(self, context, output=None):
        text  = output or ""
        score = len(text) / 5000  # 简单示例
        return SkillResult(
            skill_name  = self.name,
            success     = True,
            score       = min(score, 1.0),
            annotations = [f"文章长度: {len(text)} 字符"],
            data        = {"length": len(text)},
        )

# 挂载到 master
master = create_master(..., skills=[MySkill()])
```

## 🔧 扩展：自定义 Agent

```python
from contentforge_py.langchain_core.agents import BaseAgent
from contentforge_py.langchain_core.messages import HumanMessage

class TranslatorAgent(BaseAgent):
    id   = "translator"
    name = "翻译专家"
    icon = "🌐"

    def get_system_prompt(self):
        return "将中文文章翻译为英文，保持Markdown格式。"

    async def build_messages(self, context):
        return [HumanMessage(f"翻译：\n\n{context.get('editorResult','')}")]

# 注册到 harness 并加入 pipeline
master.harness.register(TranslatorAgent(llm=llm))
```

---

## 🧪 运行测试

```bash
python contentforge_py/examples.py test
# 期望输出: 13/13 通过 ✅
```

---

## 支持的 Provider

| Provider | 默认模型 | API Key 格式 | Base URL |
|----------|---------|-------------|---------|
| `anthropic` | claude-sonnet-4-20250514 | `sk-ant-...` | 自动 |
| `openai` | gpt-4o | `sk-...` | 自动 |
| `custom` | 自定义 | 任意 | 必填 |

---

## 技术栈

- **Python 3.8+**（仅标准库：`asyncio` · `urllib` · `http.server` · `json` · `re`）
- **零外部依赖**（LangChain 核心抽象完全自实现）
- **异步优先**（`async/await` 贯穿整个框架）
- **可测试**（所有组件可独立单元测试）
