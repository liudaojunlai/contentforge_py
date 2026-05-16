#!/usr/bin/env python3
"""
ContentForge CLI
════════════════════════════════════════════
命令行界面，支持交互式和参数模式。

用法：
  python -m contentforge_py.cli --provider anthropic --key sk-ant-xxx --request "写一篇关于AI的文章"
  python -m contentforge_py.cli --interactive

环境变量：
  LLM_PROVIDER=anthropic
  LLM_API_KEY=sk-ant-...
  LLM_MODEL=              (可选)
  LLM_BASE_URL=           (custom provider)
"""
import sys
import os
import json
import time
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contentforge_py.master import create_master, ContentRequest
from contentforge_py.langchain_core.callbacks import EventType


# ─────────────────────────────────────────────
# Terminal Colors
# ─────────────────────────────────────────────
class C:
    RST    = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    PURPLE = "\033[35m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    WHITE  = "\033[97m"
    BG_PUR = "\033[45m"

def c(text, *codes):
    return "".join(codes) + text + C.RST

def banner():
    print(c("""
╔══════════════════════════════════════════════════╗
║        ContentForge  Multi-Agent System          ║
║        LangChain · Harness · Skills v1.0         ║
╚══════════════════════════════════════════════════╝
""", C.PURPLE, C.BOLD))

def section(title):
    print(c(f"\n── {title} {'─'*(46-len(title))}", C.DIM))


# ─────────────────────────────────────────────
# Event Listener for Live Output
# ─────────────────────────────────────────────
AGENT_META = {
    "topic":    ("🎯", C.PURPLE),
    "research": ("🔬", C.CYAN),
    "outline":  ("🗂️",  C.YELLOW),
    "writer":   ("✍️",  C.GREEN),
    "editor":   ("✏️",  C.RED),
    "seo":      ("📈", "\033[33m"),
}

current_step = {"name": "", "start": 0.0}

def on_event(event):
    et = event.type
    d  = event.data

    if et == EventType.PIPELINE_START:
        print(c(f"  流水线启动，共 {len(d.get('steps',[]))} 步\n", C.DIM))

    elif et == EventType.STEP_START:
        sid  = d.get("step_id", "")
        name = d.get("name", sid)
        icon, col = AGENT_META.get(sid, ("🤖", C.WHITE))
        current_step["name"]  = name
        current_step["start"] = time.time()
        print(c(f"  {icon}  [{sid}] {name} ...", col, C.BOLD), end=" ", flush=True)

    elif et == EventType.STEP_END:
        elapsed = time.time() - current_step.get("start", time.time())
        print(c(f"✓  ({elapsed:.1f}s)", C.GREEN))

    elif et == EventType.AGENT_RETRY:
        print(c(f"\n     ↻ 重试 (attempt {d.get('attempt',1)}) ...", C.YELLOW), end=" ", flush=True)

    elif et == EventType.AGENT_ERROR:
        print(c(f"\n     ✗ {d.get('error','')[:80]}", C.RED))

    elif et == EventType.SKILL_START:
        print(c(f"     💡 Skill [{d.get('skill','')}]", C.BLUE), end=" ", flush=True)

    elif et == EventType.SKILL_END:
        print(c("✓", C.GREEN))


# ─────────────────────────────────────────────
# Output Formatters
# ─────────────────────────────────────────────
def print_result(result):
    if not result.success:
        print(c(f"\n❌ 失败: {result.error}", C.RED, C.BOLD))
        return

    outline = result.outline or {}
    seo     = result.seo     or {}
    topic   = result.topic   or {}
    skills  = result.skills  or {}

    section("📄 文章信息")
    print(f"  标题   : {c(outline.get('title', '(无标题)'), C.WHITE, C.BOLD)}")
    print(f"  副标题 : {outline.get('subtitle', '-')}")
    print(f"  角度   : {topic.get('angle', '-')}")
    print(f"  受众   : {topic.get('audience', '-')}")
    print(f"  耗时   : {c(f'{result.duration:.1f}s', C.GREEN)}")

    section("📊 SEO 报告")
    score = seo.get("seoScore", 0)
    score_col = C.GREEN if score >= 80 else C.YELLOW if score >= 60 else C.RED
    print(f"  主关键词 : {c(seo.get('primaryKeyword', '-'), C.CYAN, C.BOLD)}")
    print(f"  SEO 评分 : {c(str(score), score_col, C.BOLD)} / 100")
    print(f"  长尾关键词: {', '.join(seo.get('longtailKeywords', []))}")
    print(f"  Meta Title: {seo.get('metaTitle', '-')}")
    if seo.get("suggestions"):
        print(f"  建议:")
        for s in seo["suggestions"]:
            print(f"    • {s}")

    if skills:
        section("💡 Skill 分析")
        for skill_name, data in skills.items():
            if isinstance(data, dict):
                score_val = data.get("score")
                score_str = f"  score={score_val:.2f}" if score_val is not None else ""
                print(f"  [{skill_name}]{score_str}")
                for ann in (data.get("annotations") or [])[:3]:
                    print(f"    • {ann}")

    section("🏷️ 文章标签")
    tags = seo.get("tags", [])
    print("  " + "  ".join(f"[{t}]" for t in tags))


def save_outputs(result, output_dir: str = "."):
    """Save article MD, SEO JSON, and full report."""
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = (result.outline or {}).get("title", "article")
    slug  = "".join(c for c in title if c.isalnum() or c in " -_")[:35].strip().replace(" ", "_")
    base  = Path(output_dir) / f"{ts}_{slug}"

    saved = []

    # Article Markdown
    if result.article:
        md_path = base.with_suffix(".md")
        md_path.write_text(result.article, encoding="utf-8")
        saved.append(str(md_path))

    # SEO JSON
    if result.seo:
        seo_path = Path(str(base) + "_seo.json")
        seo_path.write_text(json.dumps(result.seo, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(str(seo_path))

    # Full report JSON
    report = {
        "topic":    result.topic,
        "outline":  result.outline,
        "research": result.research,
        "seo":      result.seo,
        "skills":   result.skills,
        "duration": result.duration,
        "logs":     result.logs,
    }
    report_path = Path(str(base) + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(str(report_path))

    return saved


# ─────────────────────────────────────────────
# Interactive Mode
# ─────────────────────────────────────────────
def interactive_mode(master):
    print(c("\n交互模式启动。输入 'quit' 退出。\n", C.CYAN))

    PLATFORMS = ["微信公众号", "知乎", "掘金", "LinkedIn", "通用媒体"]
    STYLES    = ["专业干货", "通俗易懂", "批判性分析", "故事叙述", "学术严谨"]

    while True:
        print(c("─" * 50, C.DIM))
        user_request = input(c("📝 内容需求: ", C.CYAN, C.BOLD)).strip()
        if user_request.lower() in ("quit", "exit", "q"):
            print(c("再见！", C.GREEN))
            break
        if not user_request:
            continue

        # Platform
        print(c("\n平台: ", C.DIM) + " | ".join(f"{i+1}.{p}" for i, p in enumerate(PLATFORMS)))
        p_input = input(c("选择(1-5, 默认5): ", C.DIM)).strip()
        try:
            platform = PLATFORMS[int(p_input) - 1]
        except Exception:
            platform = "通用媒体"

        # Style
        print(c("风格: ", C.DIM) + " | ".join(f"{i+1}.{s}" for i, s in enumerate(STYLES)))
        s_input = input(c("选择(1-5, 默认1): ", C.DIM)).strip()
        try:
            style = STYLES[int(s_input) - 1]
        except Exception:
            style = "专业干货"

        # Word count
        wc_input = input(c("目标字数(默认1500): ", C.DIM)).strip()
        try:
            word_count = int(wc_input)
        except Exception:
            word_count = 1500

        req = ContentRequest(
            user_request=user_request,
            platform=platform,
            style=style,
            word_count=word_count,
        )

        print(c(f"\n🚀 启动 Pipeline ...\n", C.PURPLE, C.BOLD))
        result = master.create(req)
        print_result(result)

        # Save?
        save_q = input(c("\n💾 保存文件? (y/n, 默认y): ", C.DIM)).strip().lower()
        if save_q != "n":
            saved = save_outputs(result)
            for f in saved:
                print(c(f"  ✓ {f}", C.GREEN))

        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="ContentForge — Multi-Agent Content Creation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m contentforge_py.cli --provider anthropic --key sk-ant-xxx --request "写一篇AI分析文章"
  python -m contentforge_py.cli --provider openai --key sk-xxx --interactive
  python -m contentforge_py.cli --provider custom --key xxx --base-url https://api.example.com --request "..."

Environment variables (alternative to flags):
  LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
        """
    )
    parser.add_argument("--provider",     default=os.getenv("LLM_PROVIDER", "anthropic"),
                        choices=["anthropic", "openai", "custom"],
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--key",          default=os.getenv("LLM_API_KEY", ""),
                        help="API Key")
    parser.add_argument("--model",        default=os.getenv("LLM_MODEL", ""),
                        help="Model name (optional, uses provider default)")
    parser.add_argument("--base-url",     default=os.getenv("LLM_BASE_URL", ""),
                        help="Base URL for custom provider")
    parser.add_argument("--request",  "-r", default="",
                        help="Content request (non-interactive mode)")
    parser.add_argument("--platform",     default="通用媒体",
                        help="Target platform")
    parser.add_argument("--style",        default="专业干货",
                        help="Writing style")
    parser.add_argument("--words",        default=1500, type=int,
                        help="Target word count")
    parser.add_argument("--output-dir",   default=".",
                        help="Output directory for saved files")
    parser.add_argument("--interactive",  action="store_true",
                        help="Start interactive mode")
    parser.add_argument("--no-save",      action="store_true",
                        help="Don't save output files")
    parser.add_argument("--quiet",        action="store_true",
                        help="Suppress pipeline event output")
    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    if not args.key:
        print(c("❌ 缺少 API Key！请通过 --key 参数或 LLM_API_KEY 环境变量提供。\n", C.RED, C.BOLD))
        print(c("示例：", C.DIM))
        print(c("  python -m contentforge_py.cli --provider anthropic --key sk-ant-YOUR_KEY --request '你的需求'\n", C.DIM))
        sys.exit(1)

    section("🔧 初始化")
    print(f"  Provider : {c(args.provider, C.CYAN, C.BOLD)}")
    print(f"  Model    : {args.model or '(默认)'}")
    print(f"  Skills   : word_count · readability · seo_quality · tone_analysis · format_validator")

    master = create_master(
        provider=args.provider,
        api_key=args.key,
        model=args.model,
        base_url=args.base_url,
        verbose=False,  # We handle events manually
    )

    # Attach our custom event listener (unless quiet)
    if not args.quiet:
        master.harness.bus.on("*", on_event)

    if args.interactive:
        interactive_mode(master)
        return

    if not args.request:
        print(c("❌ 请提供 --request '内容需求' 或使用 --interactive 模式\n", C.RED))
        sys.exit(1)

    req = ContentRequest(
        user_request=args.request,
        platform=args.platform,
        style=args.style,
        word_count=args.words,
    )

    section("🚀 启动 Pipeline")
    print(f"  需求: {c(args.request[:80], C.WHITE)}")
    print(f"  平台: {args.platform}  风格: {args.style}  字数: {args.words}\n")

    result = master.create(req)
    print_result(result)

    if not args.no_save and result.success:
        section("💾 保存文件")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        saved = save_outputs(result, output_dir=args.output_dir)
        for f in saved:
            print(c(f"  ✓ {f}", C.GREEN))

    print(c("\n✅ 完成！\n", C.GREEN, C.BOLD) if result.success else
          c(f"\n❌ 失败: {result.error}\n", C.RED, C.BOLD))
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
