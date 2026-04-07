"""
config/config.py — Single authoritative source of truth for all constants,
keyword registries, thresholds, and penalty configuration.

DESIGN RULES
------------
• No logic in this file — pure data declarations only.
• Every grader, reward function, and task MUST import its constants from here.
• Changing any value here propagates to the entire system automatically.
• Values are typed and documented to serve as living API documentation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment identity
# ---------------------------------------------------------------------------

ENV_NAME: str = "CustomerSupportEnv-v1"
MAX_STEPS: int = 5
EARLY_TERMINATION_THRESHOLD: float = 0.85  # apply per-task threshold below

# ---------------------------------------------------------------------------
# Reward bounds  (final scalar is always clamped to this range)
# ---------------------------------------------------------------------------

REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Penalty caps  (enforcement is in reward.py; caps documented here)
# ---------------------------------------------------------------------------

IRRELEVANCE_PENALTY_CAP: float = 0.35
REPETITION_PENALTY_CAP: float  = 0.30
STEP_PENALTY_CAP: float        = 0.20
VERBOSITY_PENALTY_CAP: float   = 0.10   # penalty for excessively long responses

# Irrelevance: overlap_ratio >= FREE_THRESHOLD → zero penalty
IRRELEVANCE_FREE_THRESHOLD: float = 0.15

# Repetition: char-4gram Jaccard >= this → penalty activates
REPETITION_SIM_THRESHOLD: float = 0.40

# Verbosity: responses > MAX_RESPONSE_WORDS get a tiny penalty
MAX_RESPONSE_WORDS: int = 120

# ---------------------------------------------------------------------------
# Task-specific score thresholds  (episode ends early if score >= threshold)
# ---------------------------------------------------------------------------

TASK_THRESHOLDS: dict[str, float] = {
    "easy":   0.88,
    "medium": 0.82,
    "hard":   0.78,
}

# ---------------------------------------------------------------------------
# Grader weight profiles per task  (must sum to 1.0 per task)
# ---------------------------------------------------------------------------

TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "easy": {
        "intent":      1.00,
        "resolution":  0.00,
        "politeness":  0.00,
        "empathy":     0.00,
    },
    "medium": {
        "intent":      0.30,
        "resolution":  0.70,
        "politeness":  0.00,
        "empathy":     0.00,
    },
    "hard": {
        "intent":      0.15,
        "resolution":  0.40,
        "politeness":  0.25,
        "empathy":     0.20,
    },
}

# ---------------------------------------------------------------------------
# Intent keyword registry
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, list[str]] = {
    "refund": [
        "refund", "money back", "return my", "reimburse", "reimbursement",
        "charge", "charged twice", "double charge", "duplicate charge",
        "overcharged", "billing error", "cancel order", "cancellation",
        "credit back", "chargeback", "payment reversed", "get my money",
        "want my money", "paid twice", "wrongly charged", "unauthorized charge",
    ],
    "delay": [
        "delay", "delayed", "late", "overdue", "still waiting",
        "not arrived", "not received", "not delivered", "missing package",
        "shipping", "shipment", "delivery", "in transit", "track",
        "tracking number", "tracking id", "order status", "where is my order",
        "where is it", "arrival", "expected date", "estimated delivery",
        "lost package", "package missing",
    ],
    "complaint": [
        "complaint", "complain", "unhappy", "dissatisfied", "frustrated",
        "terrible", "awful", "horrible", "worst", "poor service",
        "bad experience", "unacceptable", "outrageous", "appalling",
        "very disappointed", "extremely disappointed", "no response",
        "nobody cares", "ignored", "not resolved", "still no help",
        "hours on hold", "on hold", "escalate", "escalating", "escalated",
        "speak to manager", "speak to supervisor", "not satisfied",
        "quality issue", "broke", "defective", "damaged",
        # Escalation resolution terms also signal complaint recognition
        "supervisor", "manager", "senior agent", "senior specialist",
        "specialist", "priority", "connect you with",
    ],
}

INTENT_PARTIAL_SYNONYMS: dict[str, list[str]] = {
    "refund":    ["price", "cost", "fee", "invoice", "order number", "purchase"],
    "delay":     ["wait", "waiting", "slow", "pending", "processing",
                  "dispatch", "courier", "post", "mail"],
    "complaint": ["issue", "problem", "concern", "matter", "situation",
                  "experience", "feedback"],
}

# ---------------------------------------------------------------------------
# Resolution keyword registry
# ---------------------------------------------------------------------------

RESOLUTION_KEYWORDS: dict[str, list[str]] = {
    "refund_approved": [
        "refund", "refunded", "money back", "reimburs",
        "process your refund", "initiate a refund", "full refund",
        "return your payment", "credit your account", "refund initiated",
        "refund has been approved", "issue a refund", "approved your refund",
        "refund within", "business days",
    ],
    "tracking_provided": [
        "tracking number", "track your order", "tracking link",
        "tracking id", "order tracking", "here is your tracking",
        "you can track", "tracking information", "carrier",
        "check the status", "shipment status", "delivery status",
        "track at", "tracking page", "track online",
    ],
    "escalated": [
        "escalate", "escalating", "escalated",
        "supervisor", "manager", "senior agent", "senior representative",
        "specialist", "transfer your case", "escalation team",
        "higher support", "dedicated team", "priority queue",
        "connect you with", "pass this on", "raising this",
    ],
}

RESOLUTION_PARTIAL: dict[str, list[str]] = {
    "refund_approved":   ["compensation", "resolve financially", "return", "reimburse"],
    "tracking_provided": ["locate", "find your package", "investigate shipment"],
    "escalated":         ["urgent", "priority", "immediate attention", "flag"],
}

# ---------------------------------------------------------------------------
# Politeness keyword registry
# ---------------------------------------------------------------------------

POLITE_POSITIVE: list[str] = [
    "please", "sorry", "apolog", "sincerely",
    "thank you", "thanks for", "thank you for", "appreciate your",
    "we appreciate", "i appreciate", "thank you for reaching",
    "happy to help", "glad to assist", "here to help",
    "glad to help", "my pleasure", "of course",
    "certainly", "absolutely", "with pleasure",
    "be happy to", "be glad to",
    "kindly", "we value", "important to us", "we care",
    "please feel free", "do not hesitate", "let me know",
    "is there anything else", "anything else i can help",
    "anything else i can assist", "further assistance",
    "completely understand", "fully understand", "understand your",
    "rest assured", "you are right", "you re right",
    "i have approved", "we have approved", "have been approved",
    "will be processed", "has been processed", "been initiated",
    "please allow", "please be assured", "i assure you",
    "you can count on", "we are here", "we re here",
]

POLITE_NEGATIVE: list[str] = [
    "your fault", "you should have known", "obviously", "clearly you",
    "that s not my problem", "that is not my problem",
    "deal with it", "whatever", "not our fault", "not our problem",
    "stop complaining", "calm down", "you re wrong", "you are wrong",
    "read the policy", "should have read",
]

# ---------------------------------------------------------------------------
# Empathy phrase registry
# ---------------------------------------------------------------------------

EMPATHY_PHRASES: list[str] = [
    "i understand", "we understand", "i can understand",
    "i completely understand", "i totally understand",
    "i can see", "i see why", "i see how",
    "must be frustrating", "must be disappointing", "must be upsetting",
    "must feel", "that sounds frustrating", "that sounds upsetting",
    "that must be", "how frustrating",
    "i apologize", "we apologize", "i sincerely apologize",
    "we sincerely apologize", "truly sorry", "deeply sorry",
    "i m sorry", "we re sorry", "so sorry to hear",
    "sorry for the inconvenience", "sorry for the trouble",
    "sorry about this",
    "your concern", "your experience", "your situation",
    "important to you", "important to us",
    "we take this seriously", "take your concern seriously",
    "i hear you", "i hear your concern",
    "i can imagine", "this is unacceptable",
    "we sincerely", "this should not have",
    "you deserve", "you should not have",
]

# ---------------------------------------------------------------------------
# Stop-words  (excluded from content-word overlap / relevance scoring)
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    "i", "a", "an", "the", "is", "it", "to", "you", "we", "my", "your",
    "and", "or", "for", "in", "on", "of", "have", "has", "be", "am",
    "are", "was", "were", "that", "this", "with", "can", "will", "do",
    "not", "no", "yes", "at", "by", "all", "so", "as", "but", "if",
    "up", "from", "about", "into", "then", "than", "they", "he", "she",
    "would", "could", "should", "may", "might", "our", "us",
})

# ---------------------------------------------------------------------------
# Inference defaults  (consumed by inference.py; overridden by env vars)
# ---------------------------------------------------------------------------

DEFAULT_API_BASE_URL: str = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME: str   = "gpt-4o-mini"
DEFAULT_TASK_NAME: str    = "hard"
DEFAULT_SEED: int          = 42
