"""Shared fixtures for the stage 2 training and scoring tests: a byte-level tokenizer
with Qwen3's special tokens and Qwen3-8B's chat template (fixture
qwen3_8b_tokenizer_config.json), and a tiny random Qwen3 model saved beside it as a
model snapshot. torch, transformers and peft are the train dependency group; tests
that need them are skipped without it (uv run --group train pytest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
QWEN3_TOKENIZER_CONFIG = FIXTURES / "qwen3_8b_tokenizer_config.json"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
ADDED = ["<think>", "</think>"]  # Qwen3's are added tokens, not special ones


def qwen3_chat_template() -> str:
    return json.loads(QWEN3_TOKENIZER_CONFIG.read_text())["chat_template"]


def byte_tokenizer():
    """A tokenizers Tokenizer: one token per byte (byte-level BPE without merges),
    plus Qwen3's special and think tokens."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[])
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(SPECIAL)
    tokenizer.add_tokens(ADDED)
    return tokenizer


def save_tiny_snapshot(path: Path, chat_template: str | None = None) -> Path:
    """A snapshot folder as huggingface_hub downloads one: the byte tokenizer (as a
    transformers tokenizer with Qwen3-8B's chat template, or chat_template) and a
    random two-layer Qwen3, in float32."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")

    raw = byte_tokenizer()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.chat_template = chat_template or qwen3_chat_template()
    tokenizer.save_pretrained(path)
    end, text_end = raw.token_to_id("<|im_end|>"), raw.token_to_id("<|endoftext|>")
    config = transformers.Qwen3Config(
        vocab_size=raw.get_vocab_size(),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        eos_token_id=end,
        pad_token_id=text_end,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(config)
    model.generation_config = transformers.GenerationConfig(
        eos_token_id=[end, text_end], pad_token_id=text_end
    )
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_snapshot(tmp_path_factory) -> Path:
    return save_tiny_snapshot(tmp_path_factory.mktemp("snapshot"))
