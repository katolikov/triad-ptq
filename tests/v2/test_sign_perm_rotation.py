"""Phase C — Block-diagonal sign+permutation rotation tests.

Acceptance criteria from the v2 plan:
  C1. Forward equivalence cosine ≥ 0.99999 on 8 prompts × seq=256 after
      applying R = block_signed_permutation(d, G).
  C3. Exported bundle yields a bit-identical OpenCL device code object
      relative to the q4f16_1 community baseline (smoke-tested via
      `tools/verify_kernel_identity.sh` against a synthetic fixture).

Heavy SmolLM-135M / TinyLlama tests are gated behind environment
variables; the unconditional tests here run on a tiny synthetic Llama
config.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from triad_ptq._v2.rotation.sign_perm import (
    BlockRotationDiagnostics,
    apply_block_rotation_to_llama,
    block_hadamard_rotation,
    block_signed_permutation,
    build_rotation,
)


# --------------------------------------------------------------------- builders

@pytest.mark.parametrize("d, g", [(64, 32), (128, 64), (256, 32), (576, 32)])
def test_block_signed_permutation_is_orthogonal(d: int, g: int) -> None:
    Q = block_signed_permutation(d, g, seed=1)
    err = (Q.t() @ Q - torch.eye(d)).norm().item()
    assert err < 1e-5, f"orthogonality residual {err:.2e} too large"


@pytest.mark.parametrize("d, g", [(64, 32), (128, 64)])
def test_block_signed_permutation_is_one_pm_one_per_row(d: int, g: int) -> None:
    Q = block_signed_permutation(d, g, seed=7)
    counts = (Q.abs() > 0).sum(dim=1)
    assert torch.all(counts == 1), "each row must have exactly one ±1"
    counts_col = (Q.abs() > 0).sum(dim=0)
    assert torch.all(counts_col == 1), "each col must have exactly one ±1"
    nonzeros = Q[Q.abs() > 0]
    assert torch.all((nonzeros == 1) | (nonzeros == -1))


@pytest.mark.parametrize("d, g", [(64, 32), (128, 64), (256, 64)])
def test_block_hadamard_is_orthogonal(d: int, g: int) -> None:
    Q = block_hadamard_rotation(d, g, seed=3)
    err = (Q.t() @ Q - torch.eye(d)).norm().item()
    assert err < 1e-5


def test_block_signed_permutation_is_block_diagonal() -> None:
    Q = block_signed_permutation(128, 32, seed=11)
    # Off-block entries must be zero.
    for bi in range(4):
        for bj in range(4):
            if bi == bj:
                continue
            block = Q[bi * 32:(bi + 1) * 32, bj * 32:(bj + 1) * 32]
            assert torch.all(block == 0)


def test_build_rotation_dispatch() -> None:
    Q1 = build_rotation(64, 32, kind="sign_perm", seed=5)
    Q2 = build_rotation(64, 32, kind="block_hadamard", seed=5)
    assert Q1.shape == Q2.shape
    with pytest.raises(ValueError, match="unknown rotation kind"):
        build_rotation(64, 32, kind="garbage")  # type: ignore[arg-type]


def test_per_group_max_invariance_under_sign_perm() -> None:
    """Acceptance: for any vector x, |Q·x| within each group of size G is
    a permutation+sign-flip of |x| within the same group. Hence
    max(|Q·x|_g) == max(|x|_g) — group-wise max is preserved exactly."""
    torch.manual_seed(0)
    d, g = 256, 32
    x = torch.randn(d)
    Q = block_signed_permutation(d, g, seed=2)
    y = Q @ x
    for b in range(d // g):
        x_g = x[b * g:(b + 1) * g].abs()
        y_g = y[b * g:(b + 1) * g].abs()
        assert torch.isclose(x_g.max(), y_g.max(), atol=1e-6)


# --------------------------------------------------------------------- LLama walker

def _build_tiny_llama(hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4) -> torch.nn.Module:
    """Construct a HF-Llama-shaped model from random weights for fast tests.

    We avoid downloading from HF; instead we use `transformers.LlamaConfig`
    + `LlamaForCausalLM` to mint a valid topology in ~0.1 s.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model


@pytest.mark.parametrize("kind", ["sign_perm", "block_hadamard"])
def test_apply_block_rotation_to_llama_forward_equivalence(kind: str) -> None:
    """Acceptance C1: cosine ≥ 0.99999 between original and rotated forward.

    Uses a synthetic 64-dim 2-layer Llama model so the test runs in <1 s.
    Eight random token sequences of length 32 are passed through the
    original and rotated models; the LM-head logits are compared.
    """
    torch.manual_seed(0)
    model = _build_tiny_llama(hidden_size=64, num_layers=2, num_heads=4)
    # Inflate weights so RMSNorm γ is non-trivial (the fold is the most
    # error-prone step; the test must exercise a non-1.0 γ).
    for blk in model.model.layers:
        blk.input_layernorm.weight.data.uniform_(0.5, 1.5)
        blk.post_attention_layernorm.weight.data.uniform_(0.5, 1.5)
    model.model.norm.weight.data.uniform_(0.5, 1.5)

    # Capture original forward.
    inputs = torch.randint(0, 256, (8, 32))
    with torch.no_grad():
        orig_logits = model(inputs).logits.detach().clone()

    diag = apply_block_rotation_to_llama(
        model, group_size=32, kind=kind, seed=42  # type: ignore[arg-type]
    )
    assert isinstance(diag, BlockRotationDiagnostics)
    assert diag.is_block_diagonal is True
    assert diag.Q_orthogonality_err < 1e-4

    with torch.no_grad():
        rot_logits = model(inputs).logits

    # Cosine similarity, vector-flattened.
    cos = torch.nn.functional.cosine_similarity(
        orig_logits.reshape(-1), rot_logits.reshape(-1), dim=0
    ).item()
    assert cos >= 0.99999, f"forward-equivalence cosine {cos:.7f} below 0.99999"


def test_apply_block_rotation_rejects_misaligned_group_size() -> None:
    model = _build_tiny_llama(hidden_size=64, num_layers=1, num_heads=4)
    with pytest.raises(ValueError, match="not divisible by group_size"):
        apply_block_rotation_to_llama(model, group_size=24, kind="sign_perm")


# --------------------------------------------------------------------- C3 smoke

def test_verify_kernel_identity_script_passes_on_byte_identical_fixture(tmp_path: Path) -> None:
    """tools/verify_kernel_identity.sh smoke: feed the script two
    directories whose `*_devc.o` are byte-identical and assert exit 0.
    """
    script = Path("tools/verify_kernel_identity.sh").resolve()
    if not script.exists():
        pytest.skip("verify_kernel_identity.sh not present")

    v2_dir = tmp_path / "v2_bundle" / "lib"
    ref_dir = tmp_path / "ref_bundle" / "lib"
    v2_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    devc_bytes = b"FAKE_OPENCL_DEVICE_CODE_FOR_TESTS"
    (v2_dir / "model_q4f16_1_devc.o").write_bytes(devc_bytes)
    (ref_dir / "model_q4f16_1_devc.o").write_bytes(devc_bytes)

    res = subprocess.run(
        ["bash", str(script), str(v2_dir.parent), str(ref_dir.parent)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, f"script failed unexpectedly: {res.stderr}"
    assert '"match":     true' in res.stdout


def test_verify_kernel_identity_script_fails_on_byte_different_fixture(tmp_path: Path) -> None:
    script = Path("tools/verify_kernel_identity.sh").resolve()
    if not script.exists():
        pytest.skip("verify_kernel_identity.sh not present")

    v2_dir = tmp_path / "v2_bundle" / "lib"
    ref_dir = tmp_path / "ref_bundle" / "lib"
    v2_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)

    (v2_dir / "model_q4f16_1_devc.o").write_bytes(b"AAAA")
    (ref_dir / "model_q4f16_1_devc.o").write_bytes(b"BBBB")

    res = subprocess.run(
        ["bash", str(script), str(v2_dir.parent), str(ref_dir.parent)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 1, "script must fail on mismatched md5"


# --------------------------------------------------------------------- gated SmolLM

@pytest.mark.skipif(
    os.environ.get("TRIAD_RUN_SMOLLM_TESTS") != "1",
    reason="SmolLM-135M test gated; set TRIAD_RUN_SMOLLM_TESTS=1 to run",
)
def test_smollm135_forward_equivalence_under_sign_perm() -> None:
    """Heavier acceptance gate: real SmolLM-135M (hidden=576, divisible by
    G=32) must keep forward cosine ≥ 0.99999 under sign+perm.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-135M", torch_dtype=torch.float32
    ).eval()

    prompts = [
        "The capital of France is",
        "Photosynthesis converts",
        "In 1492 Columbus",
        "The fastest land animal is",
        "Quantum mechanics describes",
        "Tomatoes originated in",
        "The speed of light is",
        "The mitochondria is",
    ]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        orig = model(**enc).logits.detach().clone()

    diag = apply_block_rotation_to_llama(model, group_size=32, kind="sign_perm")
    assert diag.is_block_diagonal

    with torch.no_grad():
        rot = model(**enc).logits

    cos = torch.nn.functional.cosine_similarity(
        orig.reshape(-1), rot.reshape(-1), dim=0
    ).item()
    assert cos >= 0.99999, f"SmolLM-135M cosine {cos:.7f} below 0.99999"
