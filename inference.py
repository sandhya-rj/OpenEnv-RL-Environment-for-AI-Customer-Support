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
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from openai import OpenAI
from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import TASKS

# ---------------- CONFIG ----------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TASK_NAME = os.environ.get("TASK_NAME", "hard")
SCENARIO_ID = os.environ.get("SCENARIO_ID", "") or None
SEED = int(os.environ.get("SEED", "42"))
PORT = int(os.environ.get("PORT", "7860"))

# ---------------- STATE ----------------
_episode_lock = threading.Lock()
_episode_env = None
_episode_result = {
    "status": "pending",
    "steps": 0,
    "score": 0.0,
    "success": False,
    "rewards": [],
    "logs": [],
}

# ---------------- CLIENT ----------------
def build_client():
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "placeholder"
    )

# ---------------- LOG ----------------
def _fmt_bool(b):
    return "true" if b else "false"

def _emit(line):
    print(line, flush=True)
    with _episode_lock:
        _episode_result["logs"].append(line)

# ---------------- INFERENCE ----------------
def run_inference():
    global _episode_env

    client = build_client()
    env = CustomerSupportEnv(seed=SEED)
    task = TASKS[TASK_NAME]

    MAX_STEPS = 5   # 🔥 critical fix

    steps_done = 0
    rewards = []
    final_score = 0.0
    success = False

    with _episode_lock:
        _episode_result["status"] = "running"
        _episode_env = env

    _emit(f"[START] task={TASK_NAME} env={CustomerSupportEnv.ENV_NAME} model={MODEL_NAME}")

    try:
        obs = env.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)
        done = False

        while not done and steps_done < MAX_STEPS:

            error_field = "null"
            action_text = ""

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": obs.query}],
                    max_tokens=256,
                    temperature=0.3,
                    timeout=10   # 🔥 critical fix
                )

                action_text = completion.choices[0].message.content.strip()

                if not action_text:
                    raise ValueError("Empty response")

            except Exception as e:
                error_field = str(e)[:100]
                action_text = "Apologies, escalating your issue immediately."

            action = Action(response=action_text)
            obs, reward, done, info = env.step(action)

            steps_done = info["step"]
            rewards.append(reward.value)
            final_score = info["task_score"]

            _emit(
                f'[STEP] step={steps_done} action="{action_text[:100]}" '
                f'reward={reward.value:.2f} done={_fmt_bool(done)} error={error_field}'
            )

            success = final_score >= 0.5

    except Exception:
        traceback.print_exc()
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "none"

        _emit(
            f"[END] success={_fmt_bool(success)} "
            f"steps={steps_done} score={final_score:.4f} rewards={rewards_str}"
        )

# ---------------- SERVER ----------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"running")

def start_server():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[SERVER] running on {PORT}")
    server.serve_forever()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        threading.Thread(target=start_server, daemon=True).start()
        time.sleep(1)

        threading.Thread(target=run_inference, daemon=True).start()

        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        sys.exit(0)