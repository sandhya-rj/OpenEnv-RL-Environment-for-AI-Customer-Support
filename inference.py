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

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TASK_NAME = os.environ.get("TASK_NAME", "hard")
SCENARIO_ID = os.environ.get("SCENARIO_ID", "") or None
SEED = int(os.environ.get("SEED", "42"))
PORT = int(os.environ.get("PORT", "7860"))

MAX_STEPS = 5
OPENAI_TIMEOUT = 10
MAX_TOKENS = 256
FALLBACK_RESPONSE = "I apologize for the issue. I am escalating this to support."


def log(line: str) -> None:
    print(line, flush=True)


def build_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")


def model_to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


@dataclass
class SharedState:
    env: CustomerSupportEnv = field(default_factory=lambda: CustomerSupportEnv(seed=SEED))
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_observation: Optional[Dict[str, Any]] = None
    last_step: Optional[Dict[str, Any]] = None
    inference_finished: bool = False


STATE = SharedState()


def json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def parse_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length == 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
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
            json_response(self, 200, {"status": "ok", "service": "openenv-inference"})
            return
        json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        content_type = (self.headers.get("Content-Type") or "").lower()
        if content_type and "application/json" not in content_type:
            json_response(self, 400, {"error": "Content-Type must be application/json"})
            return

        try:
            body = parse_json_body(self)
        except Exception:
            json_response(self, 400, {"error": "invalid_json"})
            return

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
                json_response(self, 200, obs_dict)
            except Exception as exc:
                json_response(self, 500, {"error": "reset_failed", "message": str(exc)})
            return

        if self.path == "/step":
            response_text = body.get("response") or body.get("action") or FALLBACK_RESPONSE
            try:
                with STATE.lock:
                    try:
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))
                    except RuntimeError:
                        # Auto-initialize episode if /step is called before /reset.
                        STATE.env.reset(scenario_id=SCENARIO_ID, task_name=(TASK_NAME if TASK_NAME in TASKS else "hard"))
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))
                    payload = {
                        "observation": model_to_dict(obs),
                        "reward": model_to_dict(reward),
                        "done": done,
                        "info": info,
                    }
                    STATE.last_observation = payload["observation"]
                    STATE.last_step = payload
                json_response(self, 200, payload)
            except Exception as exc:
                json_response(self, 500, {"error": "step_failed", "message": str(exc)})
            return

        json_response(self, 404, {"error": "not_found"})


def run_inference() -> None:
    client = build_client()
    task_name = TASK_NAME if TASK_NAME in TASKS else "hard"

    steps = 0
    score = 0.0
    rewards = []
    success = False
    log(f"[START] task={task_name} env={CustomerSupportEnv.ENV_NAME} model={MODEL_NAME}")

    try:
        with STATE.lock:
            obs = STATE.env.reset(scenario_id=SCENARIO_ID, task_name=task_name)

        done = False
        while (not done) and steps < MAX_STEPS:
            error_msg = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": obs.query}],
                    max_tokens=MAX_TOKENS,
                    timeout=OPENAI_TIMEOUT,
                )
                text = (completion.choices[0].message.content or "").strip() or FALLBACK_RESPONSE
            except Exception as exc:
                error_msg = str(exc)
                text = FALLBACK_RESPONSE

            try:
                with STATE.lock:
                    obs, reward, done, info = STATE.env.step(Action(response=text))
                    steps = info.get("step", steps + 1)
                    score = float(info.get("task_score", 0.0))
                    rewards.append(float(reward.value))
                    STATE.last_observation = model_to_dict(obs)
                    STATE.last_step = {
                        "observation": model_to_dict(obs),
                        "reward": model_to_dict(reward),
                        "done": done,
                        "info": info,
                    }

                error_value = "null" if error_msg is None else error_msg.replace('"', "'")
                safe_action = text.replace('"', "'")[:120]
                log(
                    f'[STEP] step={steps} action="{safe_action}" reward={reward.value:.2f} '
                    f'done={str(done).lower()} error={error_value}'
                )
            except Exception as step_exc:
                safe_action = text.replace('"', "'")[:120]
                safe_step_error = str(step_exc)
                safe_step_error = safe_step_error.replace('"', "'")
                log(
                    "[STEP] step={} action=\"{}\" reward=0.00 done=false error={}".format(
                        steps,
                        safe_action,
                        safe_step_error,
                    )
                )
                break

        success = score >= 0.5
    except Exception:
        traceback.print_exc()
        success = False
    finally:
        STATE.inference_finished = True
        rewards_text = ",".join(f"{r:.2f}" for r in rewards)
        log(
            f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_text}"
        )


def start_http_server() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[SERVER] running on {PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()

    inference_thread = threading.Thread(target=run_inference, daemon=True)
    inference_thread.start()

    while True:
        time.sleep(60)