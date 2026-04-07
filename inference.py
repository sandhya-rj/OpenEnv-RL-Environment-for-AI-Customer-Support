"""
inference.py — Inference loop + HTTP server for the AI Customer Support OpenEnv environment.

Reads all configuration from environment variables:
  API_BASE_URL  — OpenAI-compatible endpoint base URL
  MODEL_NAME    — Model identifier string
  HF_TOKEN      — Bearer token (Hugging Face token or OpenAI key)
  TASK_NAME     — "easy" | "medium" | "hard"   (default: "hard")
  SCENARIO_ID   — Specific scenario ID; blank = random selection
  SEED          — Integer seed for reproducible random selection (default: 42)
  PORT          — HTTP server port (default: 7860, required by Hugging Face Spaces)

Strict log format
-----------------
  [START] task=<task_name> env=<env_name> model=<model_name>
  [STEP]  step=<n> action="<text>" reward=<0.00> done=<true|false> error=<null|msg>
  [END]   success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...>

Rules:
  • Reward formatted to exactly 2 decimal places
  • Booleans always lowercase: true / false
  • error field is always present (null when no error occurred)
  • [END] is ALWAYS printed — even when an unexpected exception occurs

HTTP API (port 7860 — Hugging Face Spaces requirement)
-------------------------------------------------------
  GET  /           → HTML status page with episode results
  GET  /health      → 200 OK  (used by HF health checks)
  POST /reset       → JSON Observation  (starts a new episode)
  POST /step        → JSON (observation, reward, done, info)
  GET  /state       → JSON EpisodeState snapshot
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import TASKS


# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str       = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str       = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN:     str       = os.environ.get("HF_TOKEN",     "")
TASK_NAME:    str       = os.environ.get("TASK_NAME",    "hard")
SCENARIO_ID:  Optional[str] = os.environ.get("SCENARIO_ID", "") or None
SEED:         int       = int(os.environ.get("SEED", "42"))
PORT:         int       = int(os.environ.get("PORT", "7860"))

# Fail-fast validation before any objects are created
if TASK_NAME not in TASKS:
    print(
        f"[ERROR] Invalid TASK_NAME={TASK_NAME!r}. Must be one of: {sorted(TASKS.keys())}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared episode state (written by inference thread, read by HTTP handlers)
# ---------------------------------------------------------------------------

_episode_lock   = threading.Lock()
_episode_env:   Optional[CustomerSupportEnv] = None   # live env for /reset + /step
_episode_result: Dict[str, Any] = {
    "status":       "pending",   # "pending" | "running" | "done" | "error"
    "task":         TASK_NAME,
    "model":        MODEL_NAME,
    "steps":        0,
    "score":        0.0,
    "success":      False,
    "rewards":      [],
    "logs":         [],          # list of [START]/[STEP]/[END] strings
    "error":        None,
}


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------

def build_client() -> OpenAI:
    """Construct an OpenAI-compatible client from environment configuration."""
    api_key = HF_TOKEN if HF_TOKEN else "placeholder-key"
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional AI customer support agent for an e-commerce company.
Your primary responsibility is to resolve customer issues promptly, accurately, and compassionately.

RESPONSE GUIDELINES:
1. EMPATHISE FIRST: Always acknowledge the customer's frustration or concern before providing a solution.
2. PROVIDE THE CORRECT RESOLUTION:
   - For billing issues or duplicate charges → initiate or confirm a full refund
   - For shipping/delivery issues → provide tracking information or investigate the shipment
   - For complaints, product quality, or escalation requests → escalate to a senior specialist or manager
3. BE POLITE AND PROFESSIONAL: Use formal, warm, respectful language at all times.
4. BE CONCISE: Do not repeat yourself or add unnecessary filler content.
5. CLOSE WITH AN OFFER: Always end by asking if there is anything else you can assist with.

Do NOT output anything except the customer-facing response text.\
"""


def build_messages(
    query: str,
    history: List[str],
    task_objective: str,
) -> List[dict]:
    """Build the messages list for the chat completion call."""
    user_parts = [f"TASK OBJECTIVE:\n{task_objective}"]

    if history:
        conv_lines = [f"[Your previous response #{i}]: {r}" for i, r in enumerate(history, 1)]
        user_parts.append("YOUR PREVIOUS RESPONSES:\n" + "\n".join(conv_lines))

    user_parts.append(f"CUSTOMER MESSAGE:\n{query}")
    user_parts.append("Please respond to the customer now.")

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": "\n\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _fmt_bool(b: bool) -> str:
    return "true" if b else "false"


def _fmt_action(text: str, max_len: int = 140) -> str:
    """Truncate and sanitise action text for the log line."""
    sanitised = text.replace("\n", " ").replace('"', "'")
    return sanitised[:max_len] + ("..." if len(sanitised) > max_len else "")


def _emit(line: str) -> None:
    """Print a log line to stdout and append to the shared result log."""
    print(line, flush=True)
    with _episode_lock:
        _episode_result["logs"].append(line)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    """Run one complete episode, emit structured logs, update shared state."""
    global _episode_env

    client    = build_client()
    env       = CustomerSupportEnv(seed=SEED)
    task      = TASKS[TASK_NAME]

    steps_done:  int        = 0
    rewards:     List[float] = []
    final_score: float      = 0.0
    success:     bool       = False

    with _episode_lock:
        _episode_result["status"] = "running"
        _episode_env = env

    start_line = f"[START] task={TASK_NAME} env={CustomerSupportEnv.ENV_NAME} model={MODEL_NAME}"
    _emit(start_line)

    try:
        obs  = env.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)
        done = False

        while not done:
            current_step: int = obs.step_number + 1
            action_text:  str = ""
            error_field:  str = "null"

            try:
                messages   = build_messages(obs.query, obs.history, task.objective)
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512,
                )
                action_text = (completion.choices[0].message.content or "").strip()
                if not action_text:
                    raise ValueError("Model returned an empty response.")

            except Exception as api_err:  # noqa: BLE001
                error_field = str(api_err).replace("\n", " ").replace('"', "'")[:200]
                # Graceful fallback — episode continues so [END] is always reached
                action_text = (
                    "I sincerely apologize for the inconvenience. "
                    "I completely understand your concern and will look into this "
                    "for you right away. Please allow me a moment to assist you."
                )

            action           = Action(response=action_text)
            obs, reward, done, info = env.step(action)

            steps_done  = info["step"]
            rewards.append(reward.value)
            final_score = info["task_score"]

            step_line = (
                f'[STEP] step={current_step} '
                f'action="{_fmt_action(action_text)}" '
                f'reward={reward.value:.2f} '
                f'done={_fmt_bool(done)} '
                f'error={error_field}'
            )
            _emit(step_line)

        success = final_score >= 0.50

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        success = False

    finally:
        # [END] — guaranteed to always print
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "none"
        end_line = (
            f"[END] success={_fmt_bool(success)} "
            f"steps={steps_done} "
            f"score={final_score:.4f} "
            f"rewards={rewards_str}"
        )
        _emit(end_line)

        with _episode_lock:
            _episode_result.update({
                "status":  "done",
                "steps":   steps_done,
                "score":   final_score,
                "success": success,
                "rewards": rewards,
            })


# ---------------------------------------------------------------------------
# HTTP server — keeps the container alive for Hugging Face Spaces
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CustomerSupportEnv — OpenEnv</title>
  <style>
    body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
    h1   {{ color: #58a6ff; }}
    pre  {{ background: #161b22; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
    .ok  {{ color: #3fb950; }} .err {{ color: #f85149; }} .info {{ color: #d29922; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    td, th {{ border: 1px solid #30363d; padding: 0.5rem 1rem; text-align: left; }}
    th {{ background: #161b22; color: #58a6ff; }}
  </style>
</head>
<body>
  <h1>🤖 CustomerSupportEnv-v1</h1>
  <p>OpenEnv RL Environment for AI Customer Support</p>
  <hr style="border-color:#30363d">
  <h2>Episode Status: <span class="{status_class}">{status}</span></h2>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>Task</td><td>{task}</td></tr>
    <tr><td>Model</td><td>{model}</td></tr>
    <tr><td>Steps</td><td>{steps}</td></tr>
    <tr><td>Final Score</td><td>{score:.4f}</td></tr>
    <tr><td>Success</td><td>{success}</td></tr>
    <tr><td>Rewards</td><td>{rewards}</td></tr>
  </table>
  <h2>Logs</h2>
  <pre>{logs}</pre>
  <h2>HTTP API</h2>
  <table>
    <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
    <tr><td>/health</td><td>GET</td><td>Health check — always returns 200</td></tr>
    <tr><td>/reset</td><td>POST</td><td>Start a new episode — returns Observation JSON</td></tr>
    <tr><td>/step</td><td>POST</td><td>Take one action — body: {{"response":"..."}} — returns (obs, reward, done, info)</td></tr>
    <tr><td>/state</td><td>GET</td><td>Current EpisodeState snapshot</td></tr>
  </table>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler — stdlib only, no new dependencies."""

    # Silence noisy access logs (keep stderr clean for real errors)
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        pass

    # ------------------------------------------------------------------ helpers

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/health", "/health/"):
            self._send_json({"status": "ok"})

        elif self.path in ("/state", "/state/"):
            global _episode_env
            with _episode_lock:
                env = _episode_env
            if env is None:
                self._send_json({"error": "No active episode. Call POST /reset first."}, 400)
                return
            try:
                state = env.state()
                self._send_json(state.model_dump())
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, 400)

        elif self.path in ("/", ""):
            with _episode_lock:
                r = dict(_episode_result)
            status      = r["status"]
            status_class = {"pending": "info", "running": "info",
                            "done":    "ok",   "error":   "err"}.get(status, "info")
            rewards_str = ", ".join(f"{v:.2f}" for v in r["rewards"]) or "none"
            logs_str    = "\n".join(r["logs"]) or "(inference pending…)"
            html = _HTML_TEMPLATE.format(
                status_class = status_class,
                status       = status.upper(),
                task         = r["task"],
                model        = r["model"],
                steps        = r["steps"],
                score        = r["score"],
                success      = str(r["success"]).lower(),
                rewards      = rewards_str,
                logs         = logs_str,
            )
            self._send_html(html)

        else:
            self._send_json({"error": f"Unknown path: {self.path}"}, 404)

    # ------------------------------------------------------------------ POST

    def do_POST(self) -> None:  # noqa: N802
        global _episode_env

        if self.path in ("/reset", "/reset/"):
            body        = self._read_json_body()
            scenario_id = body.get("scenario_id") or SCENARIO_ID
            task_name   = body.get("task_name",   TASK_NAME)

            if task_name not in TASKS:
                self._send_json({"error": f"Invalid task_name={task_name!r}"}, 400)
                return

            env = CustomerSupportEnv(seed=SEED)
            try:
                obs = env.reset(scenario_id=scenario_id, task_name=task_name)
            except (KeyError, ValueError) as exc:
                self._send_json({"error": str(exc)}, 400)
                return

            with _episode_lock:
                _episode_env = env

            self._send_json(obs.model_dump())

        elif self.path in ("/step", "/step/"):
            with _episode_lock:
                env = _episode_env

            if env is None:
                self._send_json({"error": "No active episode. Call POST /reset first."}, 400)
                return

            body     = self._read_json_body()
            response = body.get("response", "")
            if not response or not response.strip():
                self._send_json({"error": "Field 'response' must be a non-empty string."}, 400)
                return

            try:
                action              = Action(response=response)
                obs, reward, done, info = env.step(action)
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, 500)
                return

            self._send_json({
                "observation": obs.model_dump(),
                "reward":      reward.model_dump(),
                "done":        done,
                "info":        info,
            })

        else:
            self._send_json({"error": f"Unknown path: {self.path}"}, 404)


def _start_server_daemon() -> None:
    """Bind the HTTP server and serve in a daemon thread.

    Starting the server here — before the inference thread — guarantees
    that the socket is listening when Hugging Face fires its first health
    check (which can happen within seconds of container startup).
    """
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"[SERVER] Listening on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # ── STEP 1: Start HTTP server FIRST (non-negotiable for HF Spaces) ──
        # The server binds and starts accepting connections before anything
        # else. HF health checks (GET /health) will immediately get 200 OK.
        server_thread = threading.Thread(
            target=_start_server_daemon,
            daemon=True,
            name="http-server",
        )
        server_thread.start()
        print("[SERVER] HTTP server thread started.", flush=True)

        # ── STEP 2: Brief guard sleep — ensures socket is bound ─────────────
        # 1 second is sufficient for the OS to bind the port; this window
        # also lets HF fire its initial health check before inference begins.
        import time
        time.sleep(1)

        # ── STEP 3: Start inference in a background daemon thread ────────────
        inference_thread = threading.Thread(
            target=run_inference,
            daemon=True,
            name="inference",
        )
        inference_thread.start()
        print("[INFERENCE] Episode thread started.", flush=True)

        # ── STEP 4: Keep main thread alive forever ───────────────────────────
        # Both child threads are daemons — they die if the main thread exits.
        # This loop prevents that. The container will NEVER exit unless killed.
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("[MAIN] Received SIGINT — shutting down cleanly.", flush=True)
        sys.exit(0)

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
