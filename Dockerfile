# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Metadata labels ───────────────────────────────────────────────────────────
LABEL maintainer="AI Customer Support Team"
LABEL description="OpenEnv AI Customer Support RL Environment v2"
LABEL version="2.0.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/customer-support-env"

# ── Prevent Python from buffering stdout/stderr (critical for log streaming) ──
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System build dependencies ──────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (separate layer for cache efficiency) ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Configuration layer (changes least often) ─────────────────────────────────
COPY config/    ./config/
COPY openenv.yaml .

# ── Domain layer ──────────────────────────────────────────────────────────────
COPY domain/    ./domain/

# ── Utilities layer ───────────────────────────────────────────────────────────
COPY utils/     ./utils/

# ── Business logic layer ──────────────────────────────────────────────────────
COPY logic/     ./logic/

# ── Environment core layer ────────────────────────────────────────────────────
COPY core/      ./core/

# ── Tests layer ───────────────────────────────────────────────────────────────
COPY tests/     ./tests/

# ── Execution layer ───────────────────────────────────────────────────────────
COPY inference.py .

# ── Runtime environment variable defaults ─────────────────────────────────────
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV TASK_NAME="hard"
ENV SCENARIO_ID=""
ENV SEED="42"
ENV PORT="7860"

# ── Expose HTTP port (required by Hugging Face Spaces) ───────────────────────
EXPOSE 7860

# ── Health check: verify all package imports work ─────────────────────────────
RUN python -c "\
from config.config import ENV_NAME, MAX_STEPS; \
from domain.models import Action, Observation, Reward, EpisodeState; \
from domain.schemas import StepResult, EpisodeSummary; \
from utils.text_processing import normalize, tokenize; \
from utils.scoring_utils import keyword_hits, jaccard_similarity; \
from utils.validation_utils import validate_action; \
from logic.graders import grade_intent_detection, grade_resolution; \
from logic.reward import compute_reward; \
from logic.tasks import get_task, SCENARIOS, TASKS; \
from core.state_manager import StateManager; \
from core.env import CustomerSupportEnv; \
print('✓ All package imports verified.')"

# ── Run unit tests during build (regression gate) ─────────────────────────────
RUN python -m pytest tests/ -q --tb=short 2>&1 | tail -25

# ── Entry point ────────────────────────────────────────────────────────────────
CMD ["python", "inference.py"]
