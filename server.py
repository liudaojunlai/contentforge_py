#!/usr/bin/env python3
"""
ContentForge Web Server
════════════════════════════════════════════
纯标准库实现的 HTTP API 服务器（无需 FastAPI/Flask）。
提供：
  POST /api/create   — 触发内容创作流水线
  GET  /api/status   — 服务健康检查
  GET  /             — 提供 Web UI (ui/index.html)

用法：
  python -m contentforge_py.server --provider anthropic --key sk-ant-xxx --port 8080
"""
import sys
import os
import json
import time
import threading
import asyncio
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Ensure parent on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Global master agent (initialized at startup)
# ─────────────────────────────────────────────
_master = None
_config = {}


# ─────────────────────────────────────────────
# In-progress jobs store (thread-safe)
# ─────────────────────────────────────────────
import uuid

_jobs = {}       # job_id → {"status": ..., "events": [...], "result": None}
_jobs_lock = threading.Lock()


def create_job() -> str:
    jid = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[jid] = {
            "status": "running",
            "events": [],
            "result": None,
            "error":  None,
            "started_at": time.time(),
        }
    return jid


def append_event(jid: str, event_type: str, data: dict):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]["events"].append({
                "type": event_type,
                "data": data,
                "ts":   time.time(),
            })


def finish_job(jid: str, result=None, error: str = None):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]["status"]  = "done" if not error else "error"
            _jobs[jid]["result"]  = result
            _jobs[jid]["error"]   = error


def get_job(jid: str):
    with _jobs_lock:
        return dict(_jobs.get(jid, {}))


def run_pipeline_in_thread(jid: str, request_data: dict):
    """Run the pipeline in a background thread with its own event loop."""
    from contentforge_py.master import ContentRequest
    from contentforge_py.langchain_core.callbacks import EventType

    def on_event(event):
        append_event(jid, event.type.value if hasattr(event.type, 'value') else str(event.type),
                     event.data)

    # Create a fresh master with event listener
    from contentforge_py.master import create_master
    master = create_master(
        provider=_config["provider"],
        api_key=_config["api_key"],
        model=_config.get("model", ""),
        base_url=_config.get("base_url", ""),
        verbose=False,
    )
    master.harness.bus.on("*", on_event)

    req = ContentRequest(
        user_request=request_data.get("userRequest", ""),
        platform=request_data.get("platform", "通用媒体"),
        style=request_data.get("style", "专业干货"),
        word_count=int(request_data.get("wordCount", 1500)),
    )

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(master.acreate(req))
        loop.close()

        finish_job(jid, result={
            "topic":    result.topic,
            "research": result.research,
            "outline":  result.outline,
            "article":  result.article,
            "seo":      result.seo,
            "skills":   result.skills,
            "duration": result.duration,
            "agentStatuses": result.agent_statuses,
            "success":  result.success,
        })
    except Exception as e:
        finish_job(jid, error=str(e) + "\n" + traceback.format_exc()[:500])


# ─────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    log_message = lambda self, *a: None  # Silence default access logs

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, mime: str = "text/html"):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            ui_path = Path(__file__).parent / "ui" / "index.html"
            if ui_path.exists():
                self._send_file(ui_path)
            else:
                self._send_json({"error": "UI not found. Open ui/index.html directly."}, 404)

        elif path == "/api/status":
            self._send_json({
                "status": "ok",
                "version": "1.0.0",
                "provider": _config.get("provider", ""),
                "model":    _config.get("model", "(default)"),
                "skills":   ["word_count", "readability", "seo_quality", "tone_analysis", "format_validator"],
            })

        elif path.startswith("/api/job/"):
            jid = path.split("/api/job/")[-1]
            job = get_job(jid)
            if not job:
                self._send_json({"error": "Job not found"}, 404)
            else:
                self._send_json(job)

        else:
            self._send_json({"error": f"Not found: {path}"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/api/create":
            data = self._read_body()
            if not data.get("userRequest"):
                self._send_json({"error": "userRequest is required"}, 400)
                return

            jid = create_job()
            t = threading.Thread(target=run_pipeline_in_thread, args=(jid, data), daemon=True)
            t.start()
            self._send_json({"jobId": jid, "status": "running"})

        elif path == "/api/create/sync":
            # Synchronous version (blocks until done)
            data = self._read_body()
            if not data.get("userRequest"):
                self._send_json({"error": "userRequest is required"}, 400)
                return

            jid = create_job()
            run_pipeline_in_thread(jid, data)
            job = get_job(jid)
            if job.get("status") == "error":
                self._send_json({"error": job["error"]}, 500)
            else:
                self._send_json(job.get("result", {}))

        else:
            self._send_json({"error": f"Not found: {path}"}, 404)


# ─────────────────────────────────────────────
# Server Entry Point
# ─────────────────────────────────────────────
def run_server(host: str = "0.0.0.0", port: int = 8080):
    server = HTTPServer((host, port), Handler)
    print(f"\n🌐 ContentForge Server 启动")
    print(f"   本地访问: http://localhost:{port}")
    print(f"   API 文档: http://localhost:{port}/api/status")
    print(f"   Provider: {_config.get('provider')}  Model: {_config.get('model') or '(default)'}")
    print(f"\n   Ctrl+C 退出\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止。")
        server.server_close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ContentForge Web Server")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "anthropic"),
                        choices=["anthropic", "openai", "custom"])
    parser.add_argument("--key",      default=os.getenv("LLM_API_KEY", ""),
                        help="API Key")
    parser.add_argument("--model",    default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     default=8080, type=int)
    args = parser.parse_args()

    if not args.key:
        print("❌ 请提供 --key API_KEY 或设置 LLM_API_KEY 环境变量")
        sys.exit(1)

    _config.update({
        "provider": args.provider,
        "api_key":  args.key,
        "model":    args.model,
        "base_url": args.base_url,
    })

    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
