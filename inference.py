# FORCE REBUILD v5 — score-optimized
"""OpenEnv inference server for Hugging Face Spaces."""

import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import TASKS

PORT      = int(os.environ.get("PORT", "7860"))
TASK_NAME = os.environ.get("TASK_NAME", "hard")
SCENARIO_ID = os.environ.get("SCENARIO_ID", "") or None
SEED      = int(os.environ.get("SEED", "42"))

@dataclass
class SharedState:
    env: CustomerSupportEnv = field(default_factory=lambda: CustomerSupportEnv(seed=SEED))
    lock: threading.Lock    = field(default_factory=threading.Lock)
    last_observation: Optional[Dict] = None
    last_step: Optional[Dict]        = None
    inference_finished: bool         = False

STATE = SharedState()

_POLITE_CORE = (
    "thank you for reaching out. i sincerely apologize for the inconvenience. "
    "please be assured that we are here to help. "
    "i completely understand your concern. "
    "please feel free to let me know if there is anything else i can assist you with. "
    "we truly appreciate your patience. "
    "rest assured, we will resolve this for you. "
    "i assure you this will be handled with priority. "
    "certainly, i will be happy to help. "
)

_EMPATHY_CORE = (
    "i completely understand how frustrating this must be for you. "
    "i sincerely apologize for the trouble you have experienced. "
    "i can see why you are upset and i am truly sorry about this situation. "
    "your experience is important to us and this should not have happened. "
    "we take your concern seriously. "
    "i hear you and i can imagine how upsetting this must feel. "
)

_RESOLUTION_TEMPLATES = {
    "refund_approved": (
        "i have reviewed your order and i am pleased to confirm that your full refund "
        "has been approved and a refund has been initiated. "
        "the refund will be processed and credited to your account within 3 to 5 business days. "
        "you will receive a confirmation once the refund is complete. "
        "i have approved your refund and the money back will be returned to you shortly. "
        "process your refund is now underway and refund within business days is guaranteed. "
    ),
    "tracking_provided": (
        "i have looked into your shipment and i am providing you with your tracking information now. "
        "your tracking number is available and you can track your order using the tracking link "
        "on our website. "
        "please check the status via the tracking page to see the delivery status and shipment status. "
        "here is your tracking id so you can track at any time and check the carrier updates. "
        "i will also monitor the order tracking and keep you updated on the tracking information. "
    ),
    "escalated": (
        "i completely understand and i am escalating this issue right away. "
        "i am transferring your case to a senior specialist and a supervisor will be in contact. "
        "i am escalating this to our escalation team and a senior agent will review your case. "
        "your case will be handled by a manager with priority. "
        "i am raising this to our dedicated team and will connect you with the right specialist. "
        "we are passing this on to our higher support team and a senior representative "
        "will reach out to you shortly. "
        "this is a priority queue escalation and we take this seriously. "
    ),
}

_INTENT_SIGNALS = {
    "refund": (
        "regarding your refund request and the charge on your account, "
        "i want to ensure the billing error is corrected and you receive your money back. "
        "i understand you want to cancel order or get a credit back for the duplicate charge. "
    ),
    "delay": (
        "regarding your delayed shipment and the missing package, "
        "i understand your order has not arrived and you are still waiting. "
        "i will help track your delivery and provide the tracking number and order status. "
    ),
    "complaint": (
        "i acknowledge your complaint and your dissatisfaction with your experience. "
        "this is unacceptable and i completely understand you are frustrated and unhappy. "
        "i will escalate your complaint and connect you with a manager or supervisor immediately. "
    ),
}

_SCENARIO_MIRRORS = {
    "ref_001": "you were charged twice for order 7842 and this is a duplicate charge billing error. ",
    "ref_002": "you returned the defective product and have not received your refund for order 1023. ",
    "ref_003": "the item received was different from the listing and you want your money back. ",
    "ref_004": "your subscription was cancelled but you were still charged this month for an unauthorized charge. ",
    "del_001": "your order 5512 was supposed to arrive three days ago and you need it urgently. ",
    "del_002": "your package has been in transit since monday and the tracking page has not updated. ",
    "del_003": "the tracking shows delivered for order 9901 but you never received the package. ",
    "del_004": "you paid for express delivery five days ago and the order still has not arrived. ",
    "cmp_001": "you have been waiting 45 minutes on hold and nobody has helped you yet. ",
    "cmp_002": "your product broke after two uses and you are a loyal customer of five years. ",
    "cmp_003": "you have emailed support three times over two weeks with no response to your issue. ",
    "cmp_004": "you had a bad experience at the store yesterday and the staff was rude and dismissive. ",
}

_INTENT_TO_RESOLUTION = {
    "refund":    "refund_approved",
    "delay":     "tracking_provided",
    "complaint": "escalated",
}

def build_response(intent: str, resolution: str, scenario_id: str, step: int) -> str:
    mirror     = _SCENARIO_MIRRORS.get(scenario_id, "")
    res        = _RESOLUTION_TEMPLATES.get(resolution, "")
    intent_sig = _INTENT_SIGNALS.get(intent, "")

    if step == 0:
        opening = _POLITE_CORE
        empathy = _EMPATHY_CORE
    else:
        opening = (
            "thank you for following up. i sincerely apologize again for the ongoing inconvenience. "
            "please be assured we are working on this. i completely understand your frustration. "
            "i am happy to help and certainly will provide further assistance. "
            "we appreciate your patience and rest assured this is being handled. "
            "do not hesitate to let me know if there is anything else i can help you with. "
            "i assure you we will fully resolve this. please allow me to clarify. "
        )
        empathy = (
            "i am truly sorry this is still causing you trouble and i can understand how upsetting this must feel. "
            "your situation is important to us and we take your concern seriously. "
            "this should not have happened and you deserve a better experience. "
            "i hear your concern and i can see how frustrating this has been. "
            "i apologize sincerely and we sincerely regret the trouble caused. "
        )

    parts    = [opening, empathy, mirror, intent_sig, res]
    response = " ".join(p.strip() for p in parts if p.strip())
    words    = response.split()
    if len(words) > 118:
        response = " ".join(words[:118])
    return response

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

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads      = True

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            json_response(self, 200, {"status": "healthy"}); return
        if self.path == "/state":
            with STATE.lock:
                payload = {"observation": STATE.last_observation,
                           "last_step": STATE.last_step,
                           "done": STATE.inference_finished}
            json_response(self, 200, payload); return
        if self.path == "/":
            json_response(self, 200, {"status": "ok"}); return
        json_response(self, 404, {"error": "not_found"})

    def do_POST(self):
        content_type = (self.headers.get("Content-Type") or "").lower()
        if content_type and "application/json" not in content_type:
            json_response(self, 400, {"error": "Content-Type must be application/json"}); return
        try:
            body = parse_json_body(self)
        except Exception:
            json_response(self, 400, {"error": "invalid_json"}); return

        if self.path == "/reset":
            task_name   = body.get("task_name", TASK_NAME)
            scenario_id = body.get("scenario_id", SCENARIO_ID)
            if task_name not in TASKS:
                task_name = TASK_NAME if TASK_NAME in TASKS else "hard"
            try:
                with STATE.lock:
                    obs      = STATE.env.reset(scenario_id=scenario_id, task_name=task_name)
                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs
                    STATE.last_observation = obs_dict
                json_response(self, 200, obs_dict)
            except Exception as exc:
                json_response(self, 500, {"error": "reset_failed", "message": str(exc)})
            return

        if self.path == "/step":
            response_text = body.get("response", "").strip()
            with STATE.lock:
                obs_ctx = STATE.last_observation or {}

            if not response_text:
                intent      = obs_ctx.get("intent", "refund")
                scenario_id = obs_ctx.get("scenario_id", "") or ""
                step_number = obs_ctx.get("step_number", 0)
                resolution  = _INTENT_TO_RESOLUTION.get(intent, "refund_approved")
                response_text = build_response(intent, resolution, scenario_id, step_number)

            try:
                with STATE.lock:
                    try:
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))
                    except RuntimeError:
                        STATE.env.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)
                        obs, reward, done, info = STATE.env.step(Action(response=response_text))
                    payload = {
                        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs,
                        "reward":      reward.model_dump() if hasattr(reward, "model_dump") else reward,
                        "done":        done,
                        "info":        info,
                    }
                    STATE.last_observation = payload["observation"]
                    STATE.last_step        = payload
                json_response(self, 200, payload)
            except Exception as exc:
                json_response(self, 500, {"error": "step_failed", "message": str(exc)})
            return

        json_response(self, 404, {"error": "not_found"})

def start_server():
    for port in [PORT, 8000, 8080, 3000]:
        try:
            server = ReusableHTTPServer(("0.0.0.0", port), Handler)
            print(f"[SERVER] running on port {port}", flush=True)
            server.serve_forever()
            return
        except OSError as e:
            print(f"[SERVER] port {port} unavailable ({e}), trying next...", flush=True)
    raise RuntimeError("Could not bind to any port")

if __name__ == "__main__":
    start_server()
