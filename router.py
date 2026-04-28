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


# Minimal streaming chat UI served at `/`. No build step, no deps — fetches
# /v1/models on load, streams /v1/chat/completions via SSE, renders tokens
# as they arrive. System prompt + model pick + clear button.
CHAT_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qwen-engine chat</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px; font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
         background: #141414; color: #e8e8e8; max-width: 960px; margin-inline: auto; }
  header { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
  header h1 { font-size: 1rem; margin: 0; flex: 1; color: #8be9fd; }
  select, input, textarea, button {
    background: #222; color: #e8e8e8; border: 1px solid #3a3a3a;
    border-radius: 6px; padding: 8px 10px; font-size: 14px;
  }
  button { cursor: pointer; }
  button:hover:not(:disabled) { background: #2d2d2d; }
  button:disabled { opacity: 0.5; cursor: default; }
  #sys { width: 100%; min-height: 56px; resize: vertical; margin-bottom: 8px; font-family: inherit; }
  #chat { height: 64vh; overflow-y: auto; border: 1px solid #2a2a2a;
          padding: 12px; background: #0d0d0d; border-radius: 6px; }
  .msg { margin-bottom: 14px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; }
  .msg.user { color: #8be9fd; }
  .msg.user::before { content: "🧑 "; }
  .msg.assistant > .answer::before { content: "🤖 "; }
  .msg.error { color: #ff6b6b; }
  details.think { margin: 0 0 6px 0; }
  details.think > summary { cursor: pointer; color: #8a8a8a; font-size: 0.85em;
                             padding: 3px 0; user-select: none; list-style: none; }
  details.think > summary::before { content: "▸ "; color: #666; }
  details.think[open] > summary::before { content: "▾ "; }
  details.think[open] > summary { color: #bbb; }
  .think-body { color: #9a9a9a; font-size: 0.92em; border-left: 2px solid #3a3a3a;
                 padding: 4px 10px; margin: 4px 0 8px 4px; white-space: pre-wrap; }
  #row { display: flex; gap: 8px; margin-top: 10px; }
  #input { flex: 1; }
  .muted { color: #888; font-size: 12px; }
  /* markdown */
  .msg.assistant .answer { white-space: normal; }
  .msg.assistant .answer p { margin: 0 0 0.6em 0; }
  .msg.assistant .answer p:last-child { margin-bottom: 0; }
  .msg.assistant .answer ul, .msg.assistant .answer ol { margin: 0.3em 0 0.6em 1.4em; padding: 0; }
  .msg.assistant .answer li { margin: 0.15em 0; }
  .msg.assistant .answer h1, .msg.assistant .answer h2, .msg.assistant .answer h3,
  .msg.assistant .answer h4, .msg.assistant .answer h5, .msg.assistant .answer h6 {
    margin: 0.8em 0 0.3em 0; font-weight: 600; color: #f0f0f0; }
  .msg.assistant .answer h1 { font-size: 1.25em; }
  .msg.assistant .answer h2 { font-size: 1.15em; }
  .msg.assistant .answer h3 { font-size: 1.05em; }
  .msg.assistant .answer code {
    background: #2a2a2a; color: #f1c45e; padding: 1px 5px; border-radius: 4px;
    font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.9em; }
  .msg.assistant .answer pre {
    background: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 6px;
    padding: 10px 12px; overflow-x: auto; margin: 0.5em 0;
    /* ancestor .msg sets pre-wrap; override so long lines scroll instead of wrap */
    white-space: pre; word-wrap: normal; overflow-wrap: normal; max-width: 100%; }
  .msg.assistant .answer pre code {
    background: transparent; color: #e8e8e8; padding: 0; border-radius: 0;
    font-size: 0.88em; line-height: 1.45;
    white-space: pre; }
  .msg.assistant .answer blockquote {
    border-left: 3px solid #555; margin: 0.4em 0; padding: 0.1em 0 0.1em 10px; color: #c0c0c0; }
  .msg.assistant .answer table {
    border-collapse: collapse; margin: 0.5em 0; font-size: 0.93em; }
  .msg.assistant .answer th, .msg.assistant .answer td {
    border: 1px solid #3a3a3a; padding: 4px 8px; }
  .msg.assistant .answer th { background: #222; }
  .msg.assistant .answer a { color: #8be9fd; }
  .msg.assistant .answer hr { border: 0; border-top: 1px solid #333; margin: 0.8em 0; }
  .msg.assistant .answer strong { color: #f0f0f0; }
</style>
</head>
<body>
<header>
  <h1>qwen-engine</h1>
  <select id="model" title="Model"></select>
  <select id="preset" title="Qwen3.5 공식 권장 sampling">
    <option value="daily" selected>💬 일상 (general)</option>
    <option value="coding">💻 코딩 (precise)</option>
    <option value="creative">🎨 창작</option>
  </select>
  <button id="clear" title="Clear chat">🗑</button>
</header>
<textarea id="sys" placeholder="System prompt (선택). 예: 너는 한국어로만 답변하는 AI야."></textarea>
<div id="chat"></div>
<div id="row">
  <input id="input" placeholder="메시지 입력 후 Enter" autofocus>
  <button id="send">Send</button>
</div>
<div class="muted" id="status"></div>
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const sysArea = document.getElementById('sys');
const modelSel = document.getElementById('model');
const clearBtn = document.getElementById('clear');
const statusEl = document.getElementById('status');
const presetSel = document.getElementById('preset');
// Model-specific presets. 27B grid best 가 9B 에 맞지 않음:
// 9B 는 temp=1.2 에서 reasoning 에 토큰 소진, 답변 empty.
// 27B (216-req grid):
//   coding  t0.7 top_p=1.00 PP=0.0
//   daily   t1.2 top_p=0.95 PP=0.0  (9.8 최상위)
//   creative t1.2 top_p=1.00 PP=1.0
// 9B (실측, 작은 모델 reasoning 터널 빠지지 않게 temp 낮춤):
//   coding  t0.7 top_p=1.0  PP=0.0
//   daily   t0.7 top_p=1.0  PP=0.0  (temp 1.2 empty)
//   creative t1.0 top_p=1.0 PP=0.5
const PRESETS_27B = {
  // daily 를 grid 1위 temp=1.2 → 0.8 로 낮춤. grid 는 single-turn 평가라
  // 환각/일관성 반영 못 했고, 실측 멀티턴에서 temp=1.2 는 첫 턴에 환각
  // 씨앗 심고 2번째 턴에서 증폭됨. 0.8 은 tradeoff 지점 (창의성 유지하면서
  // 환각 감소).
  daily:    {temperature: 0.8, top_p: 0.95, top_k: 0, min_p: 0.0,
             presence_penalty: 0.0, repetition_penalty: 1.0},
  coding:   {temperature: 0.7, top_p: 1.0,  top_k: 0, min_p: 0.0,
             presence_penalty: 0.0, repetition_penalty: 1.0},
  creative: {temperature: 1.2, top_p: 1.0,  top_k: 0, min_p: 0.0,
             presence_penalty: 1.0, repetition_penalty: 1.0},
};
const PRESETS_9B = {
  daily:    {temperature: 0.7, top_p: 1.0,  top_k: 0, min_p: 0.0,
             presence_penalty: 0.0, repetition_penalty: 1.0},
  coding:   {temperature: 0.7, top_p: 1.0,  top_k: 0, min_p: 0.0,
             presence_penalty: 0.0, repetition_penalty: 1.0},
  creative: {temperature: 1.0, top_p: 1.0,  top_k: 0, min_p: 0.0,
             presence_penalty: 0.5, repetition_penalty: 1.0},
};
function presetFor(modelId, mode) {
  // 9b / 9B / qwen3.5-9b / Qwopus...9B 등 커버
  return /9b/i.test(modelId) ? PRESETS_9B[mode] : PRESETS_27B[mode];
}
let history = [];
let pending = false;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
// Parse a streaming/final assistant content into think + answer.
// Qwopus wraps chain-of-thought in <think>...</think> before the real
// answer. While a <think> block is still open (no closing tag yet),
// everything so far is think, answer is empty, done=false.
function splitThink(full, streamDone) {
  if (!full.startsWith('<think>')) return {think: '', answer: full, done: true};
  let rest = full.slice(7);
  if (rest.startsWith('\n')) rest = rest.slice(1);
  const ci = rest.indexOf('</think>');
  if (ci < 0) {
    // Still streaming → show as in-progress reasoning.
    // Stream ended without </think> (Qwopus skip-close) → treat as answer.
    if (streamDone) return {think: '', answer: full, done: true};
    return {think: rest, answer: '', done: false};
  }
  const think = rest.slice(0, ci);
  let ans = rest.slice(ci + 8).replace(/^[\s\n\r]+/, '');
  return {think, answer: ans, done: true};
}
function renderMsg(m) {
  const wrap = el('div', 'msg ' + m.role);
  if (m.role === 'assistant') {
    const {think, answer, done} = splitThink(m.content, !!m.streamDone);
    if (think) {
      const d = document.createElement('details');
      d.className = 'think';
      // default: open while streaming the reasoning, closed once the
      // answer starts. User toggles persist via m.thinkOpen override.
      const autoOpen = !done;
      d.open = (m.thinkOpen !== undefined) ? m.thinkOpen : autoOpen;
      d.addEventListener('toggle', () => { m.thinkOpen = d.open; });
      const label = done ? '사고 과정' : '생각중…';
      d.appendChild(el('summary', '', `${label} (${think.length}자)`));
      d.appendChild(el('div', 'think-body', think));
      wrap.appendChild(d);
    }
    const ansDiv = el('div', 'answer');
    // Render markdown for assistant answers. `marked` loads via CDN; if it
    // fails (offline) we fall back to plain text so the chat still works.
    if (typeof marked !== 'undefined') {
      try {
        marked.setOptions({breaks: true, gfm: true});
        ansDiv.innerHTML = marked.parse(answer || '');
      } catch (e) {
        ansDiv.textContent = answer;
      }
    } else {
      ansDiv.textContent = answer;
    }
    wrap.appendChild(ansDiv);
  } else {
    wrap.textContent = m.content;
  }
  return wrap;
}
function render() {
  // Only autoscroll when the user is pinned to the absolute bottom.
  // Any upward scroll — even a few px — disables autoscroll until they
  // return to the bottom.
  const atBottom = (chat.scrollHeight - chat.scrollTop - chat.clientHeight) <= 2;
  chat.innerHTML = '';
  for (const m of history) chat.appendChild(renderMsg(m));
  if (atBottom) chat.scrollTop = chat.scrollHeight;
}
async function loadModels() {
  try {
    const r = await fetch('/v1/models');
    const d = await r.json();
    modelSel.innerHTML = '';
    for (const m of d.data) {
      const o = el('option'); o.value = m.id; o.textContent = m.id;
      modelSel.appendChild(o);
    }
    // Prefer 9b alias as default (faster)
    const prefer = d.data.find(m => /9b/i.test(m.id));
    if (prefer) modelSel.value = prefer.id;
  } catch (e) {
    statusEl.textContent = 'models load fail: ' + e.message;
  }
}
async function send() {
  if (pending) return;
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  history.push({role: 'user', content: text});
  const asst = {role: 'assistant', content: ''};
  history.push(asst);
  render();

  const msgs = [];
  const sys = sysArea.value.trim();
  if (sys) msgs.push({role: 'system', content: sys});
  for (const m of history.slice(0, -1)) msgs.push(m);  // exclude pending assistant stub

  pending = true;
  sendBtn.disabled = true;
  statusEl.textContent = 'streaming…';
  const t0 = performance.now();
  let tokenCount = 0;
  let tFirst = 0;
  let tickTimer = null;
  const updateTick = () => {
    if (!tFirst) return;
    const now = performance.now();
    const gen_s = (now - tFirst) / 1000;
    const tps = gen_s > 0 ? tokenCount / gen_s : 0;
    statusEl.textContent = `streaming · ttft ${((tFirst - t0)).toFixed(0)}ms · ${tokenCount} tok · ${tps.toFixed(1)} tok/s`;
  };
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: modelSel.value,
        messages: msgs,
        stream: true,
        max_tokens: 4096,
        ...presetFor(modelSel.value, presetSel.value)
      })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    tickTimer = setInterval(updateTick, 250);
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const ln of lines) {
        const t = ln.trim();
        if (!t.startsWith('data:')) continue;
        const payload = t.slice(5).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          const j = JSON.parse(payload);
          const delta = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
          if (delta) {
            if (!tFirst) tFirst = performance.now();
            asst.content += delta;
            tokenCount++;
            render();
          }
        } catch {}
      }
    }
    asst.streamDone = true;
    render();
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    const total_s = (performance.now() - t0) / 1000;
    const ttft_ms = tFirst ? (tFirst - t0) : 0;
    const gen_s = tFirst ? (performance.now() - tFirst) / 1000 : 0;
    const tps = gen_s > 0 ? tokenCount / gen_s : 0;
    statusEl.textContent = `done · ttft ${ttft_ms.toFixed(0)}ms · ${tokenCount} tok · ${tps.toFixed(1)} tok/s · total ${total_s.toFixed(1)}s`;
  } catch (e) {
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    asst.content += '\n[error: ' + e.message + ']';
    asst.streamDone = true;
    render();
    statusEl.textContent = 'error';
  }
  pending = false;
  sendBtn.disabled = false;
  input.focus();
}
sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
clearBtn.addEventListener('click', () => { history = []; render(); statusEl.textContent = ''; });
loadModels();
</script>
</body>
</html>
"""


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
        if self.path in ("/v1/models", "/api/tags", "/api/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": self.registry.list_models()})
        elif self.path == "/":
            body = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            for k, v in CORS.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.registry.refresh()
            self._send_json(200, {
                "status": "ok",
                "aliases": list(self.registry.aliases.keys()),
                "upstream_ids": list(self.registry.upstream_ids.keys()),
            })
        else:
            self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/api/chat",
                             "/v1/completions", "/api/v1/chat/completions",
                             "/api/v1/completions"):
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
        elif getattr(self, "upstream_api_key", None):
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"

        # PocketPal (and some other Ollama-ish clients) prefix OpenAI routes
        # with /api/. Upstream qwen-engine only knows /v1/... so strip the
        # leading /api to match. /api/chat stays as-is (native Ollama route).
        upstream_path = self.path
        if upstream_path.startswith("/api/v1/"):
            upstream_path = upstream_path[len("/api"):]
        target = f"{upstream}{upstream_path}"
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
        # Background-fed queue so the main thread can interleave SSE comment
        # heartbeats while the upstream is silent (long prefill, model
        # thinking). Without heartbeats, Tailscale / corporate proxies drop
        # idle TCP after 15-60s and the client gets a "network error".
        import queue, threading
        q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=64)

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

            def feeder():
                try:
                    for chunk in r.iter_raw():
                        if chunk:
                            q.put(chunk)
                finally:
                    q.put(None)  # sentinel: upstream done

            t = threading.Thread(target=feeder, daemon=True)
            t.start()

            heartbeat_interval = 10.0  # seconds
            while True:
                try:
                    chunk = q.get(timeout=heartbeat_interval)
                except queue.Empty:
                    # Idle: send SSE comment as keepalive. Spec: lines
                    # starting with ":" are comments, ignored by clients.
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                if chunk is None:
                    return  # upstream done
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
    ap.add_argument("--upstream-api-key", default=None,
                    help="API key to inject as Authorization: Bearer <key> when "
                         "the inbound request has no Authorization header. "
                         "Lets clients hit the router without knowing the upstream key.")
    args = ap.parse_args()
    Handler.upstream_api_key = args.upstream_api_key

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
