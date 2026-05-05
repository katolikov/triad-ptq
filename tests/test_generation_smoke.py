"""Generation smoke test.

Catches PTQ failures where eval-window PPL looks fine but the model
generates degenerate output (repeating tokens, ungrammatical loops).

Motivated by ADR-007: clip_search lowered TinyLlama-1.1B's WikiText-2
PPL by 0.13 but the deployed v2 model emitted "the-l-i-c- and the
the-l-i-c-..." indefinitely. A small 4-gram repetition detector on a
SmolLM-135M-INT4 fixture catches that class of failure on host before
device deployment.

By default the test SKIPS if the fixture is not built (keeps `make
test` fast). To activate the regression guard locally or in CI:

    TRIAD_BUILD_FIXTURES=1 uv run pytest tests/test_generation_smoke.py

The first run builds the fixture under /tmp/triad-test-fixtures/
(~60 s on M1); subsequent runs reuse it.
"""
from __future__ import annotations

import pytest
import torch

from triad_ptq.testing import (
    load_smollm135_quantized_int4,
    smollm135_fixture_available,
)


GENERATION_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):",
]


@pytest.mark.skipif(
    not smollm135_fixture_available(),
    reason="SmolLM-135M INT4 fixture not built; run with TRIAD_BUILD_FIXTURES=1",
)
def test_generation_does_not_degenerate():
    """Quantized model must not produce repeating-token degenerate output."""
    model, tokenizer = load_smollm135_quantized_int4()
    model.eval()

    for prompt in GENERATION_PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0])
        new_text = text[len(prompt):]

        # 4-gram repetition detector: any 4-token substring that
        # appears 3+ times in the 32-token output is a collapse signal.
        words = new_text.split()
        for i in range(len(words) - 4):
            ngram = tuple(words[i : i + 4])
            count = sum(
                1 for j in range(len(words) - 4)
                if tuple(words[j : j + 4]) == ngram
            )
            assert count < 3, (
                f"Degenerate generation on '{prompt}': "
                f"4-gram {ngram} repeats {count} times in: {new_text!r}"
            )
