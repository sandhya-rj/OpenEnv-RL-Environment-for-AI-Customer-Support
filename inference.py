"""
inference.py — Inference loop for the AI Customer Support OpenEnv environment.

Reads all configuration from environment variables:
  API_BASE_URL  — OpenAI-compatible endpoint base URL
  MODEL_NAME    — Model identifier string
  HF_TOKEN      — Bearer token (Hugging Face token or OpenAI key)
  TASK_NAME     — "easy" | "medium" | "hard"   (default: "hard")
  SCENARIO_ID   — Specific scenario ID; blank = random selection
  SEED          — Integer seed for reproducible random selection (default: 42)

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
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List, Optional

from openai import OpenAI

from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import TASKS


# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
TASK_NAME: str = os.environ.get("TASK_NAME", "hard")
SCENARIO_ID: Optional[str] = os.environ.get("SCENARIO_ID", "") or None
SEED: int = int(os.environ.get("SEED", "42"))

# Fail-fast validation before any objects are created
if TASK_NAME not in TASKS:
    print(
        f"[ERROR] Invalid TASK_NAME={TASK_NAME!r}. Must be one of: {sorted(TASKS.keys())}",
        file=sys.stderr,
    )
    sys.exit(1)


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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
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


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    """Run one complete episode and emit structured logs."""
    client = build_client()
    env = CustomerSupportEnv(seed=SEED)
    task = TASKS[TASK_NAME]

    steps_done: int = 0
    rewards: List[float] = []
    final_score: float = 0.0
    success: bool = False

    # [START] — always the first line
    print(
        f"[START] task={TASK_NAME} env={CustomerSupportEnv.ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )

    try:
        obs = env.reset(scenario_id=SCENARIO_ID, task_name=TASK_NAME)
        done = False

        while not done:
            current_step: int = obs.step_number + 1
            action_text: str = ""
            error_field: str = "null"

            try:
                messages = build_messages(obs.query, obs.history, task.objective)
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
                # Graceful fallback — episode continues to [END] is always reached
                action_text = (
                    "I sincerely apologize for the inconvenience. "
                    "I completely understand your concern and will look into this "
                    "for you right away. Please allow me a moment to assist you."
                )

            action = Action(response=action_text)
            obs, reward, done, info = env.step(action)

            steps_done = info["step"]
            rewards.append(reward.value)
            final_score = info["task_score"]

            # [STEP] — one line per action
            print(
                f'[STEP] step={current_step} '
                f'action="{_fmt_action(action_text)}" '
                f'reward={reward.value:.2f} '
                f'done={_fmt_bool(done)} '
                f'error={error_field}',
                flush=True,
            )

        success = final_score >= 0.50

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        success = False

    finally:
        # [END] — guaranteed to always print
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "none"
        print(
            f"[END] success={_fmt_bool(success)} "
            f"steps={steps_done} "
            f"score={final_score:.4f} "
            f"rewards={rewards_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_inference()
