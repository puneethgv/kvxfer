"""The closed-form solver must recover a known linear map from moments alone.

Every quantity the solver uses -- the fit, the intercept, R² -- is derived from
accumulated second moments rather than from samples. These tests check that
against ground truth on synthetic data, where the true map is known, so that a
mistake in the moment algebra cannot hide behind plausible-looking numbers on
real activations.
"""

from __future__ import annotations

import pytest
import torch

from kvxfer.geometry import KVGeometry
from kvxfer.solvers.ridge import held_out_r2, rank_source_layers, solve_ridge
from kvxfer.stats import GramAccumulator

KV_DIM = 16
N_SRC_LAYERS = 4
N_TGT_LAYERS = 2


def _geom(n_layers: int, kv_dim: int = KV_DIM) -> KVGeometry:
    return KVGeometry(
        model_id="synthetic",
        n_layers=n_layers,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=kv_dim // 2,
        hidden_size=64,
        rope_theta=1e6,
    )


def _fit_synthetic(n_tokens=4000, noise=0.0, seed=0, chunks=5):
    """Build stats for Y = X W + b (+ noise) with a known W, in several chunks."""
    torch.manual_seed(seed)
    dim_in = N_SRC_LAYERS * KV_DIM
    w_true = torch.randn(dim_in, KV_DIM, dtype=torch.float64) / dim_in**0.5
    b_true = torch.randn(KV_DIM, dtype=torch.float64)

    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    per_chunk = n_tokens // chunks
    for _ in range(chunks):
        x = torch.randn(per_chunk, dim_in, dtype=torch.float64)
        y0 = x @ w_true + b_true
        if noise:
            y0 = y0 + noise * torch.randn_like(y0)
        # Target layer 1 is pure noise: an unpredictable control.
        y = torch.stack([y0, torch.randn_like(y0)], dim=1)
        acc.update(x, y)

    return acc.finalize(), w_true, b_true


def test_recovers_exact_linear_map():
    """With noiseless data and a tiny penalty, the fit must match ground truth."""
    stats, w_true, b_true = _fit_synthetic(noise=0.0)
    layers = tuple(range(N_SRC_LAYERS))

    fit = solve_ridge(stats, layers, target_layer=0, lam=1e-10)

    assert torch.allclose(fit.weight.double(), w_true, atol=1e-4), (
        f"weight error {(fit.weight.double() - w_true).abs().max():.2e}"
    )
    assert torch.allclose(fit.bias.double(), b_true, atol=1e-4)
    assert fit.r2 > 0.999, f"noiseless fit should be near-perfect, got R2={fit.r2:.4f}"


def test_intercept_recovered_without_centering_pass():
    """A large offset must be absorbed by the bias, not the weights.

    Centering is done on the moments, so this specifically tests that the
    rank-one correction is right.
    """
    stats, w_true, b_true = _fit_synthetic(noise=0.0)
    fit = solve_ridge(stats, tuple(range(N_SRC_LAYERS)), 0, lam=1e-10)
    assert torch.allclose(fit.bias.double(), b_true, atol=1e-4)


def test_r2_is_near_zero_for_unpredictable_target():
    """The control target is independent noise; R² must not claim signal."""
    stats, _, _ = _fit_synthetic(noise=0.0)
    fit = solve_ridge(stats, tuple(range(N_SRC_LAYERS)), target_layer=1, lam=1e-6)
    assert abs(fit.r2) < 0.05, f"found spurious structure in noise: R2={fit.r2:.4f}"


def test_r2_tracks_noise_level():
    """More observation noise must lower R² monotonically."""
    scores = []
    for noise in (0.0, 0.25, 1.0):
        stats, _, _ = _fit_synthetic(noise=noise)
        scores.append(solve_ridge(stats, tuple(range(N_SRC_LAYERS)), 0, lam=1e-8).r2)
    assert scores[0] > scores[1] > scores[2], f"R2 not monotone in noise: {scores}"


def test_stronger_penalty_shrinks_weights():
    """Ridge must actually regularize: larger lambda, smaller norm."""
    stats, _, _ = _fit_synthetic(noise=0.5)
    layers = tuple(range(N_SRC_LAYERS))
    norms = [
        solve_ridge(stats, layers, 0, lam=lam).weight.norm().item()
        for lam in (1e-8, 1e-2, 1.0)
    ]
    assert norms[0] > norms[1] > norms[2], f"weights not shrinking: {norms}"


def test_layer_selection_finds_the_informative_layers():
    """Ranking must prefer source layers that actually carry signal."""
    torch.manual_seed(1)
    dim_in = N_SRC_LAYERS * KV_DIM
    # Only source layers 1 and 3 contribute to the target.
    w_true = torch.zeros(dim_in, KV_DIM, dtype=torch.float64)
    for layer in (1, 3):
        w_true[layer * KV_DIM : (layer + 1) * KV_DIM] = torch.randn(
            KV_DIM, KV_DIM, dtype=torch.float64
        ) / KV_DIM**0.5

    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    for _ in range(5):
        x = torch.randn(1000, dim_in, dtype=torch.float64)
        y0 = x @ w_true
        acc.update(x, torch.stack([y0, torch.randn_like(y0)], dim=1))
    stats = acc.finalize()

    ranking = rank_source_layers(stats, target_layer=0, lam=1e-8)
    top_two = {layer for layer, _ in ranking[:2]}
    assert top_two == {1, 3}, f"selection picked {top_two}, expected {{1, 3}}"


def test_select_submatrix_matches_direct_fit():
    """Fitting on a layer subset via submatrix extraction must equal the
    result of having accumulated only those layers in the first place.

    This is the assumption the whole full-Gram design rests on: that layer
    selection is free after one calibration pass.
    """
    torch.manual_seed(2)
    dim_in = N_SRC_LAYERS * KV_DIM
    subset = (0, 2)

    x_chunks = [torch.randn(800, dim_in, dtype=torch.float64) for _ in range(3)]
    w_true = torch.randn(dim_in, KV_DIM, dtype=torch.float64) / dim_in**0.5

    full = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    partial = GramAccumulator(
        _geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys", source_layers=subset
    )
    cols = torch.cat([torch.arange(l * KV_DIM, (l + 1) * KV_DIM) for l in subset])

    for x in x_chunks:
        y0 = x @ w_true
        y = torch.stack([y0, torch.randn_like(y0)], dim=1)
        full.update(x, y)
        partial.update(x[:, cols], y)

    from_full = solve_ridge(full.finalize(), subset, 0, lam=1e-6)
    from_partial = solve_ridge(partial.finalize(), subset, 0, lam=1e-6)

    assert torch.allclose(from_full.weight, from_partial.weight, atol=1e-6)
    assert from_full.r2 == __import__("pytest").approx(from_partial.r2, abs=1e-9)


def test_held_out_r2_matches_direct_residual_computation():
    """Held-out R² from moments must equal the value computed from samples.

    The moment expansion is easy to get subtly wrong in the bias terms, and a
    wrong validation score would silently corrupt every selection decision.
    """
    import pytest

    torch.manual_seed(7)
    dim_in = N_SRC_LAYERS * KV_DIM
    w_true = torch.randn(dim_in, KV_DIM, dtype=torch.float64) / dim_in**0.5
    b_true = torch.randn(KV_DIM, dtype=torch.float64)

    fit_stats, _, _ = _fit_synthetic(n_tokens=6000, noise=0.3, seed=7)
    fit = solve_ridge(fit_stats, tuple(range(N_SRC_LAYERS)), 0, lam=1e-3)

    # An independent validation split, kept as samples so we can score directly.
    x_val = torch.randn(3000, dim_in, dtype=torch.float64)
    y_val = x_val @ w_true + b_true + 0.3 * torch.randn(3000, KV_DIM, dtype=torch.float64)

    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    acc.update(x_val, torch.stack([y_val, torch.randn_like(y_val)], dim=1))
    val_stats = acc.finalize()

    from_moments = held_out_r2(val_stats, fit)

    pred = x_val @ fit.weight.double() + fit.bias.double()
    rss = ((y_val - pred) ** 2).sum(0)
    tss = ((y_val - y_val.mean(0)) ** 2).sum(0)
    direct = 1.0 - rss / tss

    assert torch.allclose(from_moments, direct, atol=1e-6), (
        f"moment-based R2 differs from direct by "
        f"{(from_moments - direct).abs().max():.2e}"
    )


def test_in_sample_r2_is_misleading_when_underdetermined():
    """With fewer tokens than parameters, in-sample R² saturates at 1.

    This is why selection is done out of sample: an in-sample criterion cannot
    distinguish between candidates in the regime the mapper actually operates
    in.
    """
    stats, _, _ = _fit_synthetic(n_tokens=N_SRC_LAYERS * KV_DIM // 2, noise=1.0, chunks=1)
    fit = solve_ridge(stats, tuple(range(N_SRC_LAYERS)), 0, lam=1e-10)
    assert fit.r2 > 0.99, (
        "expected in-sample R2 to saturate when the design interpolates; "
        f"got {fit.r2:.4f}"
    )


def test_check_pair_rejects_models_that_do_not_share_a_tokenizer():
    """Matched KV geometry is necessary but not sufficient.

    One set of token ids is prefilled through both models, so they must agree
    on what those ids mean. Mistral-7B-v0.3 and Ministral-8B have identical KV
    geometry -- 8 heads of 128 -- and vocabularies of 32,768 and 131,072. That
    pair passed every check, ran a full pipeline on rented hardware, and
    produced chance-level accuracy on every condition routed through the
    target, while the source, scored with its own tokenizer, looked healthy.
    """
    from kvxfer.geometry import IncompatiblePairError, KVGeometry, check_pair

    def geom(model_id: str, vocab: int) -> KVGeometry:
        return KVGeometry(
            model_id=model_id, n_layers=32, n_q_heads=32, n_kv_heads=8,
            head_dim=128, hidden_size=4096, rope_theta=1e6, vocab_size=vocab,
        )

    check_pair(geom("a", 131072), geom("b", 131072))  # same tokenizer: fine

    with pytest.raises(IncompatiblePairError, match="do not share a tokenizer"):
        check_pair(geom("small-vocab", 32768), geom("big-vocab", 131072))
