"""Qwen3's chat layout with thinking off, as vLLM sends it, and Qwen3-8B's serving
limits (PLAN.md Decisions 2026-11-01).

vLLM renders a request's messages with the model's chat template and
chat_template_kwargs {"enable_thinking": false}, adds the generation prompt (the
assistant turn opening with an empty think block) and tokenizes the text as it is
(no special tokens added). The model then writes the reply and ends it with
<|im_end|>. Training uses the same text: prompt_text is what vLLM tokenizes,
target_text what the model should generate. secbench.train checks both against the
model's own template for every example, and secbench.score checks the token ids
against vLLM's /tokenize before scoring.

Library only.
"""

from __future__ import annotations

IM_END = "<|im_end|>"
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
THINKING_OFF = {"enable_thinking": False}  # chat_template_kwargs

# vLLM's --max-model-len for Qwen3-8B (README), and the output tokens scoring allows:
# a prompt longer than MAX_MODEL_LEN - MAX_TOKENS doesn't fit.
MAX_MODEL_LEN = 40960
MAX_TOKENS = 1024


def prompt_text(system: str, user: str) -> str:
    """The system and user messages and the generation prompt, as vLLM renders them
    with thinking off."""
    return (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user}{IM_END}\n"
        f"{GENERATION_PROMPT}"
    )


def target_text(reply: str) -> str:
    """What the model generates after prompt_text: the reply, then <|im_end|>, where
    generation stops."""
    return reply + IM_END
