# server.py — HTTP environment server
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import TASKS

PORT      = int(os.environ.get("PORT", "7860"))
TASK_NAME = os.environ.get("TASK_NAME", "hard")
SCENARIO_ID = os.environ.get("SCENARIO_ID", "") or None
SEED      = int(os.environ.get("SEED", "42"))

print("[SERVER] Loading environment...", flush=True)
_ENV  = CustomerSupportEnv(seed=SEED)
_ENV.reset(task_name=TASK_NAME)
print("[SERVER] Environment ready.", flush=True)

_LOCK      = threading.Lock()
_LAST_OBS: Optional[Dict] = None
_LAST_STEP: Optional[Dict] = None

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
    return json.loads(handler.rfile.read(length).decode("utf-8"))

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads      = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): return

    def do_GET(self):
        if self.path == "/health":
            json_response(self, 200, {"status": "healthy"}); return
        if self.path == "/state":
            with _LOCK:
                json_response(self, 200, {"observation": _LAST_OBS, "last_step": _LAST_STEP, "done": False})
            return
        if self.path == "/":
            json_response(self, 200, {"status": "ok"}); return
        json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        global _LAST_OBS, _LAST_STEP
        ct = (self.headers.get("Content-Type") or "").lower()
        if ct and "application/json" not in ct:
            json_response(self, 400, {"error": "Content-Type must be application/json"}); return
        try:
            body = parse_json_body(self)
        except Exception:
            json_response(self, 400, {"error": "invalid_json"}); return

        if self.path == "/reset":
            task_name   = body.get("task_name", TASK_NAME)
            scenario_id = body.get("scenario_id", SCENARIO_ID)
            if task_name not in TASKS:
                task_name = "hard"
            try:
                with _LOCK:
                    obs      = _ENV.reset(scenario_id=scenario_id, task_name=task_name)
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
                    _LAST_OBS = obs_dict
                json_response(self, 200, obs_dict)
            except Exception as exc:
                json_response(self, 500, {"error": "reset_failed", "message": str(exc)})
            return

        if self.path == "/step":
            response_text = body.get("response", "").strip()
            if not response_text:
                response_text = "I will help you with your request."
            try:
                with _LOCK:
                    try:
                        obs, reward, done, info = _ENV.step(Action(response=response_text))
                    except RuntimeError:
                        _ENV.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)
                        obs, reward, done, info = _ENV.step(Action(response=response_text))
                    payload = {
                        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
                        "reward":      reward.model_dump() if hasattr(reward, "model_dump") else reward,
                        "done":        done,
                        "info":        info,
                    }
                    _LAST_OBS  = payload["observation"]
                    _LAST_STEP = payload
                json_response(self, 200, payload)
            except Exception as exc:
                json_response(self, 500, {"error": "step_failed", "message": str(exc)})
            return
        json_response(self, 404, {"error": "not_found"})

if __name__ == "__main__":
    for port in [PORT, 8000, 8080, 3000]:
        try:
            server = ReusableHTTPServer(("0.0.0.0", port), Handler)
            print(f"[SERVER] running on port {port}", flush=True)
            server.serve_forever()
            break
        except OSError as e:
            print(f"[SERVER] port {port} unavailable ({e})", flush=True)
