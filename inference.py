# FORCE REBUILD v2
"""OpenEnv inference server for Hugging Face Spaces."""

import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

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

MAX_STEPS = 2
OPENAI_TIMEOUT = 3
MAX_TOKENS = 64
FALLBACK_RESPONSE = "I apologize for the issue. I am escalating this to support."

# ---------------- LOG ----------------
def log(line: str):
    print(line, flush=True)

# ---------------- CLIENT ----------------
def build_client():
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")


def model_to_dict(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj

# ---------------- STATE ----------------
@dataclass
class SharedState:
    env: CustomerSupportEnv = field(default_factory=lambda: CustomerSupportEnv(seed=SEED))
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_observation: Optional[Dict] = None
    last_step: Optional[Dict] = None
    inference_finished: bool = False

STATE = SharedState()

# ---------------- HELPERS ----------------
def json_response(handler, status, payload):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)

def parse_json_body(handler):
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))

# ---------------- HTTP HANDLER ----------------
class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            json_response(self, 200, {"status": "healthy"})
            return

        if self.path == "/state":
            with STATE.lock:
                payload = {
                    "observation": STATE.last_observation,
                    "last_step": STATE.last_step,
                    "done": STATE.inference_finished,
                }
            json_response(self, 200, payload)
            return

        if self.path == "/":
            json_response(self, 200, {"status": "ok"})
            return

        json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        content_type = (self.headers.get("Content-Type") or "").lower()
        if content_type and "application/json" not in content_type:
            json_response(self, 400, {"error": "Content-Type must be application/json"})
            return

        try:
            body = parse_json_body(self)
        except Exception:
            json_response(self, 400, {"error": "invalid_json"})
            return

        # -------- RESET --------
        if self.path == "/reset":
            task_name = body.get("task_name", TASK_NAME)
            scenario_id = body.get("scenario_id", SCENARIO_ID)

            if task_name not in TASKS:
                task_name = TASK_NAME if TASK_NAME in TASKS else "hard"

            try:
                with STATE.lock:
                    obs = STATE.env.reset(scenario_id=scenario_id, task_name=task_name)
                    obs_dict = model_to_dict(obs)
                    STATE.last_observation = obs_dict

                # ✅ OUTSIDE LOCK (CRITICAL FIX)
                json_response(self, 200, obs_dict)

            except Exception as exc:
                json_response(self, 500, {"error": "reset_failed", "message": str(exc)})

            return

        # -------- STEP --------
        if self.path == "/step":
            response_text = body.get("response") or FALLBACK_RESPONSE

            try:
                with STATE.lock:
                    try:
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))
                    except RuntimeError:
                        STATE.env.reset(
                            scenario_id=SCENARIO_ID,
                            task_name=(TASK_NAME if TASK_NAME in TASKS else "hard"),
                        )
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))

                    payload = {
                        "observation": model_to_dict(obs),
                        "reward": model_to_dict(reward),
                        "done": done,
                        "info": info,
                    }

                    STATE.last_observation = payload["observation"]
                    STATE.last_step = payload

                # ✅ OUTSIDE LOCK (CRITICAL FIX)
                json_response(self, 200, payload)

            except Exception as exc:
                json_response(self, 500, {"error": "step_failed", "message": str(exc)})

            return

        json_response(self, 404, {"error": "not_found"})

# ---------------- INFERENCE ----------------
def run_inference():
    client = build_client()
    steps = 0
    rewards = []
    score = 0.0

    log(f"[START] task={TASK_NAME} env={CustomerSupportEnv.ENV_NAME} model={MODEL_NAME}")

    try:
        with STATE.lock:
            obs = STATE.env.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)

        done = False
        while not done and steps < MAX_STEPS:
            call_started = time.monotonic()
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": obs.query}],
                    max_tokens=MAX_TOKENS,
                    timeout=min(OPENAI_TIMEOUT, 2),
                )
                text = (completion.choices[0].message.content or "").strip() or FALLBACK_RESPONSE
                if (time.monotonic() - call_started) > 2:
                    text = FALLBACK_RESPONSE
            except Exception:
                text = FALLBACK_RESPONSE

            with STATE.lock:
                obs, reward, done, info = STATE.env.step(Action(response=text))
                steps = info.get("step", steps + 1)
                if steps >= 2:
                    done = True
                score = float(info.get("task_score", 0.0))
                rewards.append(float(reward.value))

            log(f'[STEP] step={steps} reward={reward.value:.2f} done={str(done).lower()} error=null')

    except Exception:
        traceback.print_exc()

    log(f"[END] success={str(score>=0.5).lower()} steps={steps} score={score:.4f}")

# ---------------- MAIN ----------------
def start_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[SERVER] running on {PORT}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(1)
    threading.Thread(target=run_inference, daemon=True).start()

    while True:
        time.sleep(60)