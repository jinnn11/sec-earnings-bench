import pytest
from conftest import byte_tokenizer, qwen3_chat_template

from secbench import chat, labels

MESSAGES = [
    {"role": "system", "content": "You read a release.\n\nReply with JSON."},
    {"role": "user", "content": "# Results\n\n| Revenue | $1,234 |"},
    {"role": "assistant", "content": '{"period_end": "2025-06-30"}'},
]
PROMPT = (
    "<|im_start|>system\nYou read a release.\n\nReply with JSON.<|im_end|>\n"
    "<|im_start|>user\n# Results\n\n| Revenue | $1,234 |<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def test_layout() -> None:
    assert chat.prompt_text(MESSAGES[0]["content"], MESSAGES[1]["content"]) == PROMPT
    assert chat.target_text(MESSAGES[2]["content"]) == (
        '{"period_end": "2025-06-30"}<|im_end|>'
    )


def test_labels_counted_the_same_layout() -> None:
    system, user, reply = (m["content"] for m in MESSAGES)
    assert labels.chat_text(MESSAGES) == (
        chat.prompt_text(system, user) + chat.target_text(reply) + "\n"
    )


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    """Qwen3-8B's chat template (pinned revision) on a byte-level tokenizer, rendered by
    transformers as vLLM renders it."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=byte_tokenizer(), eos_token="<|im_end|>"
    )
    tokenizer.chat_template = qwen3_chat_template()
    return tokenizer


def test_the_prompt_is_qwen3s_generation_prompt_with_thinking_off(
    qwen3_tokenizer,
) -> None:
    # vLLM passes a request's chat_template_kwargs as keywords.
    rendered = qwen3_tokenizer.apply_chat_template(
        MESSAGES[:2], tokenize=False, add_generation_prompt=True, **chat.THINKING_OFF
    )
    assert rendered == PROMPT


def test_the_target_ends_qwen3s_conversation(qwen3_tokenizer) -> None:
    system, user, reply = (m["content"] for m in MESSAGES)
    rendered = qwen3_tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, **chat.THINKING_OFF
    )
    assert rendered == chat.prompt_text(system, user) + chat.target_text(reply) + "\n"


def test_thinking_on_is_another_prompt(qwen3_tokenizer) -> None:
    """The check can fail: without enable_thinking false there is no empty think
    block, so the model would be trained on a prompt it never sees."""
    rendered = qwen3_tokenizer.apply_chat_template(
        MESSAGES[:2], tokenize=False, add_generation_prompt=True
    )
    assert rendered == PROMPT.removesuffix("<think>\n\n</think>\n\n")


def test_special_tokens_are_single_tokens() -> None:
    """The layout's markers are tokens of their own, as in Qwen3's tokenizer."""
    tokenizer = byte_tokenizer()
    ids = tokenizer.encode(PROMPT, add_special_tokens=False).ids
    start, end = (tokenizer.token_to_id(t) for t in ("<|im_start|>", "<|im_end|>"))
    assert ids.count(start) == 3 and ids.count(end) == 2
    think, end_think = (tokenizer.token_to_id(t) for t in ("<think>", "</think>"))
    [newline] = tokenizer.encode("\n").ids
    assert ids[-6:] == [think, newline, newline, end_think, newline, newline]
