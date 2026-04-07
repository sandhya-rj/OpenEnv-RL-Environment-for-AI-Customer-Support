# CustomerSupportEnv — OpenEnv RL Environment for AI Customer Support

> A **production-grade reinforcement learning environment** where an AI agent learns to resolve real-world customer support interactions — accurately, empathetically, and efficiently.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://openenv.dev)
[![Tests](https://img.shields.io/badge/tests-143%20passed-success)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

**CustomerSupportEnv-v1** is a fully [OpenEnv](https://openenv.dev)-compliant RL environment that simulates AI-driven customer support interactions in an e-commerce context. The agent receives a customer message, generates a natural-language response, and is scored across four quality dimensions: **intent detection**, **resolution correctness**, **politeness**, and **empathy**.

Every episode provides a **dense, continuous reward signal** at each step — making it suitable for direct policy training, LLM fine-tuning (RLHF/DPO), and structured evaluation of frontier models.

---

## Why This Matters

Real-world customer support is one of the highest-volume, highest-stakes NLP deployment contexts:

- Fortune 500 companies handle **millions of support tickets per month**
- Response quality directly impacts **customer retention and revenue**
- Current LLMs lack structured, graded feedback for support-specific behaviour

This environment solves the **grading problem**: it provides a deterministic, continuous, multi-signal evaluation framework that can train agents to produce responses that are simultaneously correct, professional, and empathetic — properties that binary metrics cannot capture.

| Domain | Concrete Use Case |
|---|---|
| E-commerce | Auto-resolve refund, delay, and complaint tickets |
| Banking | Deterministic policy evaluation for dispute resolution |
| Telecom | KPI-driven complaint resolution fine-tuning |
| SaaS | RLHF reward modelling for support LLMs |
| Research | Benchmark for customer-facing dialogue RL agents |

---

## Key Features

- **6 deterministic partial-credit graders** — no randomness, no binary pass/fail
- **Continuous reward in `[-1.0, 1.0]`** at every step — not sparse, not end-of-episode
- **3 task difficulty tiers** (easy → medium → hard) with calibrated thresholds
- **12 multi-turn scenarios** across 3 intent types and 2 sentiment levels
- **4 penalty signals**: irrelevance, repetition, step cost, verbosity
- **OpenEnv-compliant API**: `reset()` / `step()` / `state()` with typed Pydantic models
- **Production-ready architecture**: fully modular, no circular imports, immutable state
- **143 passing unit + integration tests** covering all public API surface

---

## Environment Design

### Observation Space

What the agent sees at each step:

```python
class Observation(BaseModel):
    query:        str        # Current customer message (may change to follow-up after step 1)
    sentiment:    str        # "happy" | "neutral" | "angry"
    history:      List[str]  # Ordered list of all prior agent responses this episode
    step_number:  int        # 0-based step index (0 on first reset())
    intent:       str        # "refund" | "delay" | "complaint" | "unknown"
    episode_done: bool       # True when the episode has terminated
```

### Action Space

What the agent produces at each step:

```python
class Action(BaseModel):
    response: str   # Natural-language reply to the customer (non-empty, non-blank)
```

The `response` field is validated on construction — blank or whitespace-only strings raise a `ValueError`.

### State Transitions

```
reset()  →  initial Observation (step=0, history=[], episode_done=False)
  │
step()   →  (Observation, Reward, done, info)
  │           ├─ history gains the agent response
  │           ├─ query switches to follow-up at step 1 (if episode continues)
  │           ├─ done=True when step >= max_steps (5) OR task_score >= threshold
  │           └─ state() returns current EpisodeState snapshot at any time
```

All state transitions are **immutable** — `StateManager.advance()` always returns a new `EpisodeState` object; the previous state is never mutated.

---

## Task System

### 🟢 Easy — Intent Detection

**Goal**: Explicitly name the customer's intent category in the response.

- Grader: `grade_intent_detection` only
- Threshold for early termination: **0.88**

| Signal | Weight |
|---|---|
| Intent detection | 100% |

### 🟡 Medium — Correct Resolution

**Goal**: Apply the correct resolution action and acknowledge the issue type.

- Grader: `grade_resolution` (70%) + `grade_intent_detection` (30%)
- Threshold: **0.82**

| Intent | Required Resolution |
|---|---|
| `refund` | Initiate / approve a full refund |
| `delay` | Provide tracking number / shipment status |
| `complaint` | Escalate to specialist or manager |

### 🔴 Hard — Full Response Quality

**Goal**: Correct resolution + professional politeness + genuine empathy, simultaneously.

- Grader: 4-signal weighted composite
- Threshold: **0.78**

| Signal | Weight |
|---|---|
| Resolution correctness | 40% |
| Politeness | 25% |
| Empathy | 20% |
| Intent identification | 15% |

---

## Reward Design

### Formula

```
reward = task_score − irrelevance_penalty − repetition_penalty − step_penalty − verbosity_penalty
```

All values clamped to **`[-1.0, 1.0]`**.

### Components

| Component | Range | Trigger |
|---|---|---|
| `task_score` | `[0, 1]` | Weighted grader composite — varies by task |
| `irrelevance_penalty` | `[0, 0.35]` | Content-word overlap with query < 15% |
| `repetition_penalty` | `[0, 0.30]` | 4-gram Jaccard similarity to prior turn > 40% |
| `step_penalty` | `[0, 0.20]` | Smooth logistic curve after step 2 when unresolved |
| `verbosity_penalty` | `[0, 0.10]` | Response exceeds 120 words |

### Why It Is Not Binary

Every grader uses **smooth, multi-band scoring** with partial credit:

- `grade_intent_detection`: keyword density ratio + contamination penalty → `[0.0, 1.0]`
- `grade_resolution`: 3-band scoring (0–15% / 15–40% / 40%+) → `[0.0, 1.0]`
- `grade_politeness`: diminishing-returns positive signals minus rude-phrase penalty → `[0.0, 1.0]`
- `grade_empathy`: diminishing-returns empathy phrase scoring with length ceiling → `[0.0, 1.0]`

All penalties are **smooth curves**, not step functions. A response that is *slightly* irrelevant receives a small penalty; one that is completely off-topic receives the cap. This gives a learning signal at every quality level.

### Penalty Design Rationale

| Penalty | Why It Exists |
|---|---|
| Irrelevance | Prevents content-free boilerplate responses from earning reward |
| Repetition | Prevents copy-paste policies in multi-turn episodes |
| Step cost | Incentivises resolving the issue efficiently (shorter episodes rewarded) |
| Verbosity | Prevents keyword-stuffing via padding |

---

## Scenario Catalogue

12 scenarios across 3 intents and 2 sentiments. Each includes a **follow-up query** surfaced at step 2 to simulate multi-turn dialogue:

| ID | Intent | Sentiment | Correct Resolution |
|---|---|---|---|
| `ref_001` | refund | angry | refund_approved |
| `ref_002` | refund | neutral | refund_approved |
| `ref_003` | refund | neutral | refund_approved |
| `ref_004` | refund | angry | refund_approved |
| `del_001` | delay | angry | tracking_provided |
| `del_002` | delay | neutral | tracking_provided |
| `del_003` | delay | angry | tracking_provided |
| `del_004` | delay | angry | tracking_provided |
| `cmp_001` | complaint | angry | escalated |
| `cmp_002` | complaint | angry | escalated |
| `cmp_003` | complaint | angry | escalated |
| `cmp_004` | complaint | angry | escalated |

---

## Architecture

```
CustomerSupportEnv-v1/
│
├── config/                   ← CONFIGURATION (imports by all layers)
│   └── config.py               Constants, keywords, thresholds — single source of truth
│
├── domain/                   ← DOMAIN MODELS
│   ├── models.py               Pydantic v2: Observation, Action, Reward, EpisodeState
│   └── schemas.py              External schemas: StepResult, EpisodeSummary
│
├── utils/                    ← UTILITIES (no project imports)
│   ├── text_processing.py      normalize, tokenize, char_ngrams, content_tokens
│   ├── scoring_utils.py        keyword_hits, char_jaccard, diminishing_sum, overlap_ratio
│   └── validation_utils.py     validate_action, validate_task_name, env guards
│
├── logic/                    ← BUSINESS LOGIC
│   ├── graders.py              6 deterministic graders + GraderOutput dataclass
│   ├── reward.py               compute_reward(), combine_grader_scores(), RewardResult
│   └── tasks.py                EasyTask, MediumTask, HardTask + 12-scenario catalogue
│
├── core/                     ← ENVIRONMENT
│   ├── env.py                  OpenEnv orchestration: reset() / step() / state()
│   └── state_manager.py        Immutable episode state transitions
│
├── tests/                    ← TEST SUITE (143 tests)
│   ├── conftest.py             Shared fixtures
│   ├── test_env.py             Integration tests for env API
│   ├── test_graders.py         Unit tests for all 6 graders
│   └── test_tasks.py           Unit tests for task registry + scenario catalogue
│
├── inference.py              ← INFERENCE LOOP ([START]/[STEP]/[END] format)
├── openenv.yaml              ← OPENENV MANIFEST
├── Dockerfile                ← python:3.10-slim, test gate in build
└── requirements.txt
```

**Dependency order** (no circular imports):

```
config → domain → utils → logic → core → inference
```

---

## OpenEnv API

```python
from core.env import CustomerSupportEnv
from domain.models import Action

env = CustomerSupportEnv(seed=42)

# 1. Start a new episode
obs = env.reset(scenario_id="ref_001", task_name="hard")

# 2. Step through the episode
action = Action(response=(
    "I completely understand how upsetting a duplicate charge must be. "
    "I sincerely apologize. I have approved a full refund for your order, "
    "which will be processed within 3–5 business days."
))
obs, reward, done, info = env.step(action)

# 3. Inspect reward breakdown
print(f"reward        = {reward.value:.4f}")   # [-1, 1]
print(f"task_score    = {info['task_score']:.4f}")
print(f"resolution    = {info['resolution_score']:.4f}")
print(f"politeness    = {info['politeness_score']:.4f}")
print(f"empathy       = {info['empathy_score']:.4f}")
print(f"penalty       = {info['penalty']:.4f}")    # always <= 0
print(f"done          = {done}")

# 4. Inspect full episode state
state = env.state()   # → EpisodeState
print(env.render())   # human-readable debug view
```

---

## Inference Script

`inference.py` runs a complete episode using any OpenAI-compatible endpoint and emits structured log lines.

### Log Format (strict)

```
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP]  step=<n> action="<text>" reward=<0.00> done=<true|false> error=<null|message>
[END]   success=<true|false> steps=<n> score=<0.0000> rewards=<r1,r2,...>
```

**Guarantees**:
- `[END]` always prints — even on unexpected exceptions (via `finally:` block)
- Reward formatted to exactly **2 decimal places**
- Booleans always lowercase: `true` / `false`
- `error` field always present — `null` when no error occurred
- All print calls use `flush=True` for container log streaming

### Actual Inference Output (gpt-4o-mini, task=hard, scenario=ref_001)

```
[START] task=hard env=CustomerSupportEnv-v1 model=gpt-4o-mini
[STEP] step=1 action="I completely understand how frustrating a duplicate charge must be, and I sincerely apologize for the inconvenience. I have approved a full ..." reward=0.53 done=false error=null
[STEP] step=2 action="The refund for your order has been approved and will appear in your account within 3-5 business days. Thank you for your patience and please..." reward=0.36 done=false error=null
[END] success=false steps=2 score=0.3625 rewards=0.53,0.36
```

> When `HF_TOKEN` or `API_BASE_URL` is unavailable, the script uses a professional fallback response and continues — `[END]` is always printed.

---

## Baseline Results

Scores measured deterministically (`seed=42`, `scenario_id=ref_001`, one step).

### Best-case response (comprehensive, empathetic, correct resolution)

| Task | `task_score` | `intent_score` | `resolution_score` | `politeness_score` | `empathy_score` |
|---|---|---|---|---|---|
| Easy | **0.7905** | 0.7905 | — | — | — |
| Medium | **0.5871** | 0.7905 | 0.5000 | — | — |
| Hard | **0.5267** | 0.7905 | 0.5000 | 0.4420 | 0.4880 |

### Score sensitivity across response quality levels (Hard task)

| Response Type | `task_score` | `reward` | Notes |
|---|---|---|---|
| Perfect (empathetic + correct) | 0.5267 | +0.5267 | No penalties |
| Partially correct | 0.3028 | +0.3028 | Keyword coverage < 40% |
| Irrelevant (off-topic) | 0.0000 | −0.3500 | Max irrelevance penalty |
| Repeated (same as prior turn) | 0.5267 | +0.2062 | Repetition penalty −0.30 |
| Verbose (keyword-padded) | 0.5329 | +0.4929 | Verbosity penalty −0.04 |

The reward is **continuously variable** — no constant outputs, no binary values.

---

## Setup

### Requirements

```
pydantic>=2.0,<3
openai>=1.10.0
PyYAML>=6.0
pytest>=7.4.0
```

### Install

```bash
pip3 install -r requirements.txt
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | _(empty)_ | Bearer token (HF or OpenAI key) |
| `TASK_NAME` | `hard` | `easy` \| `medium` \| `hard` |
| `SCENARIO_ID` | _(empty)_ | Specific scenario; blank = random |
| `SEED` | `42` | Integer seed for reproducible selection |

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export TASK_NAME="hard"
export SEED="42"

python3 inference.py
```

### Run Tests

```bash
# Full suite (143 tests)
python3 -m pytest tests/ -v

# Individual suites
python3 -m pytest tests/test_env.py -v      # 46 environment tests
python3 -m pytest tests/test_graders.py -v  # 55 grader unit tests
python3 -m pytest tests/test_tasks.py -v    # 42 task/scenario tests
```

Expected output:

```
============================= 143 passed in 0.09s ==============================
```

---

## Docker

```bash
# Build — runs all 143 tests as a build-time regression gate
docker build -t support-env .

# Run inference (pass env vars at runtime)
docker run \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_..." \
  -e TASK_NAME="hard" \
  -e SEED="42" \
  support-env

# Override to a specific scenario
docker run -e HF_TOKEN="hf_..." -e SCENARIO_ID="ref_001" support-env
```

**Build-time gates** (both must pass before `CMD` is reached):
1. Import health check — all packages verified importable
2. `python3 -m pytest tests/ -q` — all 143 tests must pass

---

## Reproducibility

| Mechanism | Guarantee |
|---|---|
| `seed` parameter on `CustomerSupportEnv(seed=N)` | Deterministic random scenario selection |
| `SEED` environment variable | Reproducible inference runs |
| Graders are purely lexical / rule-based | Same input always produces same score |
| No ML models in the scoring pipeline | Zero stochasticity in reward computation |
| Pydantic v2 frozen models (`Reward`) | Reward cannot be mutated after construction |

---

## OpenEnv Validation

```bash
openenv validate openenv.yaml
```

The `openenv.yaml` manifest fully specifies the observation space, action space, reward space, task definitions, scenario catalogue, penalty configuration, and episode termination conditions.

---

## Limitations & Future Work

| Limitation | Planned Improvement |
|---|---|
| Keyword-based graders (lexical only) | Replace with embedding-similarity scoring for semantic coverage |
| 12 fixed scenarios | Procedural scenario generation from templates |
| Single-turn follow-up per episode | Full multi-turn dialogue trees |
| English only | Multilingual customer query support |
| E-commerce domain only | Financial, telecom, and healthcare scenario packs |

---

## Module Reference

| Module | Package | Responsibility |
|---|---|---|
| `config.py` | `config/` | All constants, keywords, thresholds — single truth source |
| `models.py` | `domain/` | Pydantic v2 types: Observation, Action, Reward, EpisodeState |
| `schemas.py` | `domain/` | External schemas: StepResult, EpisodeSummary |
| `text_processing.py` | `utils/` | normalize, tokenize, char_ngrams, content_tokens |
| `scoring_utils.py` | `utils/` | keyword_hits, char_jaccard, diminishing_sum, overlap_ratio |
| `validation_utils.py` | `utils/` | Input guards — action, task name, scenario, env state |
| `graders.py` | `logic/` | 6 deterministic graders + GraderOutput dataclass |
| `reward.py` | `logic/` | compute_reward(), combine_grader_scores(), RewardResult |
| `tasks.py` | `logic/` | EasyTask / MediumTask / HardTask + 12-scenario catalogue |
| `state_manager.py` | `core/` | Immutable episode state transitions |
| `env.py` | `core/` | OpenEnv orchestration layer (reset / step / state) |
| `inference.py` | root | Structured inference loop with [START]/[STEP]/[END] |

---

## License

MIT
