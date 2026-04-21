"""OpenAI-compatible router: single port fan-in for qwen-engine backends.

Run:
    python3 router.py --port 8000 \
        --backend qwen3.5-27b=http://127.0.0.1:8080 \
        --backend qwen3.5-9b=http://127.0.0.1:8081

Request model field accepts any of:
    - alias given on CLI (e.g. "qwen3.5-27b")
    - the upstream's own model id (as reported by its /v1/models)
    - substring match against alias or upstream id (case-insensitive)

/v1/models returns the merged list from all backends plus aliases.
SSE streaming is passed through without buffering.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


class Registry:
    def __init__(self, backends: dict[str, str]):
        self.aliases = backends
        self.upstream_ids: dict[str, str] = {}
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=httpx.Timeout(10.0, read=None))
        self.refresh()

    def refresh(self):
        new_ids = {}
        for alias, url in self.aliases.items():
            try:
                r = self.client.get(f"{url}/v1/models", timeout=3.0)
                for m in r.json().get("data", []):
                    mid = m.get("id")
                    if mid:
                        new_ids[mid] = url
            except Exception as e:
                print(f"[router] warn: {alias} ({url}) /v1/models failed: {e}",
                      file=sys.stderr)
        with self.lock:
            self.upstream_ids = new_ids

    def resolve(self, model: str) -> str | None:
        if not model:
            if len(self.aliases) == 1:
                return next(iter(self.aliases.values()))
            return None
        if model in self.aliases:
            return self.aliases[model]
        with self.lock:
            if model in self.upstream_ids:
                return self.upstream_ids[model]
            needle = model.lower()
            for alias, url in self.aliases.items():
                if needle in alias.lower():
                    return url
            for mid, url in self.upstream_ids.items():
                if needle in mid.lower():
                    return url
        return None

    def list_models(self) -> list[dict]:
        out = []
        seen = set()
        for alias in self.aliases:
            if alias not in seen:
                out.append({"id": alias, "object": "model", "owned_by": "router"})
                seen.add(alias)
        with self.lock:
            for mid in self.upstream_ids:
                if mid not in seen:
                    out.append({"id": mid, "object": "model", "owned_by": "local"})
                    seen.add(mid)
        return out


CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


class Handler(BaseHTTPRequestHandler):
    registry: Registry = None  # set by main

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[router] {self.address_string()} {fmt % args}\n")

    def _send_json(self, status: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/api/tags"):
            self._send_json(200, {"object": "list", "data": self.registry.list_models()})
        elif self.path in ("/", "/health"):
            self.registry.refresh()
            self._send_json(200, {
                "status": "ok",
                "aliases": list(self.registry.aliases.keys()),
                "upstream_ids": list(self.registry.upstream_ids.keys()),
            })
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/api/chat", "/v1/completions"):
            self._send_json(404, {"error": {"message": "Not found"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw)
        except Exception:
            self._send_json(400, {"error": {"message": "Invalid JSON"}})
            return

        model = req.get("model", "")
        upstream = self.registry.resolve(model)
        if not upstream:
            self._send_json(400, {"error": {
                "message": f"Unknown model: {model!r}. Available: "
                           f"{list(self.registry.aliases.keys())}"}})
            return

        stream = bool(req.get("stream", False))
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

        target = f"{upstream}{self.path}"
        try:
            if stream:
                self._proxy_stream(target, headers, raw)
            else:
                self._proxy_blocking(target, headers, raw)
        except httpx.HTTPError as e:
            self._send_json(502, {"error": {"message": f"Upstream error: {e}"}})

    def _proxy_blocking(self, target, headers, body):
        r = self.registry.client.post(target, headers=headers, content=body,
                                      timeout=httpx.Timeout(10.0, read=None))
        payload = r.content
        self.send_response(r.status_code)
        self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(payload)))
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _proxy_stream(self, target, headers, body):
        with self.registry.client.stream("POST", target, headers=headers,
                                         content=body,
                                         timeout=httpx.Timeout(10.0, read=None)) as r:
            self.send_response(r.status_code)
            self.send_header("Content-Type", r.headers.get("Content-Type",
                                                           "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            for k, v in CORS.items():
                self.send_header(k, v)
            self.end_headers()
            for chunk in r.iter_raw():
                if not chunk:
                    continue
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return


def parse_backend(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected alias=url, got {spec!r}")
    alias, url = spec.split("=", 1)
    alias = alias.strip()
    url = url.strip().rstrip("/")
    if not alias or not url:
        raise argparse.ArgumentTypeError(f"bad --backend {spec!r}")
    return alias, url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--backend", action="append", type=parse_backend, default=[],
                    help="alias=url, repeatable")
    args = ap.parse_args()

    if not args.backend:
        args.backend = [
            ("qwen3.5-27b", "http://127.0.0.1:8080"),
            ("qwen3.5-9b", "http://127.0.0.1:8081"),
        ]

    backends = dict(args.backend)
    Handler.registry = Registry(backends)

    print(f"[router] aliases: {list(backends.keys())}")
    print(f"[router] upstream ids: {list(Handler.registry.upstream_ids.keys())}")
    print(f"[router] listening on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
