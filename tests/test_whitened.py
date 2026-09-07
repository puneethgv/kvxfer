"""The attention-aligned solver and the metrics it minimizes against.

The load-bearing property is reduction: with an identity metric the whitened
solve must reproduce plain ridge to numerical precision. If it does not, any
difference measured later between the two is an artifact of the implementation
rather than of the objective, and the entire comparison is void.
"""

from __future__ import annotations

import pytest
import torch

from kvxfer.geometry import KVGeometry
from kvxfer.metrics import (
    HeadMetrics,
    identity_metrics,
    key_metric,
    rope_frequencies,
)
from kvxfer.solvers.ridge import solve_ridge
from kvxfer.solvers.whitened import clear_design_cache, solve_whitened
from kvxfer.stats import GramAccumulator

N_KV_HEADS = 2
HEAD_DIM = 8
KV_DIM = N_KV_HEADS * HEAD_DIM
N_SRC_LAYERS = 3
N_TGT_LAYERS = 2


def _geom(n_layers: int) -> KVGeometry:
    return KVGeometry(
        model_id="synthetic",
        n_layers=n_layers,
        n_q_heads=N_KV_HEADS * 2,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        hidden_size=64,
        rope_theta=1e6,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_design_cache()
    yield
    clear_design_cache()


def _stats(noise: float = 0.4, seed: int = 0, n_tokens: int = 6000):
    torch.manual_seed(seed)
    dim_in = N_SRC_LAYERS * KV_DIM
    w_true = torch.randn(dim_in, KV_DIM, dtype=torch.float64) / dim_in**0.5
    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    for _ in range(4):
        x = torch.randn(n_tokens // 4, dim_in, dtype=torch.float64)
        y0 = x @ w_true + noise * torch.randn(n_tokens // 4, KV_DIM, dtype=torch.float64)
        acc.update(x, torch.stack([y0, torch.randn_like(y0)], dim=1))
    return acc.finalize()


def test_identity_metric_reduces_to_plain_ridge():
    """With M = I the whitened solve must equal the isotropic solve.

    This is the control for every later comparison between the two objectives.
    """
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    metrics = identity_metrics(_geom(N_TGT_LAYERS), "keys")

    plain = solve_ridge(stats, layers, 0, lam=1e-2)
    whitened = solve_whitened(
        stats, layers, 0, metrics, N_KV_HEADS, HEAD_DIM, lam=1e-2
    )

    assert torch.allclose(plain.weight, whitened.weight, atol=1e-6), (
        f"max weight difference {(plain.weight - whitened.weight).abs().max():.2e}"
    )
    assert torch.allclose(plain.bias, whitened.bias, atol=1e-6)
    assert whitened.r2 == pytest.approx(plain.r2, abs=1e-9)


def test_uniform_metric_scaling_is_a_no_op():
    """Scaling the metric by a constant must not change the fit.

    Only the *relative* weighting of directions is meaningful; an overall scale
    would otherwise silently act as a second, hidden penalty setting.
    """
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    base = identity_metrics(_geom(N_TGT_LAYERS), "keys")
    scaled = HeadMetrics(base.matrices * 37.0, "keys").normalized()

    a = solve_whitened(stats, layers, 0, base, N_KV_HEADS, HEAD_DIM, lam=1e-2)
    b = solve_whitened(stats, layers, 0, scaled, N_KV_HEADS, HEAD_DIM, lam=1e-2)

    assert torch.allclose(a.weight, b.weight, atol=1e-8)


def test_metric_shifts_fit_toward_weighted_directions():
    """A metric that emphasizes some coordinates must fit those better.

    Downweighted directions are shrunk harder, so the whitened fit should beat
    plain ridge on the emphasized coordinates and lose on the ignored ones --
    that trade is the entire point of the objective.
    """
    stats = _stats(noise=0.8)
    layers = tuple(range(N_SRC_LAYERS))

    # Emphasize the first half of each head's coordinates by 100x.
    matrices = identity_metrics(_geom(N_TGT_LAYERS), "keys").matrices.clone()
    emphasized = list(range(HEAD_DIM // 2))
    for j in emphasized:
        matrices[:, :, j, j] = 100.0
    metrics = HeadMetrics(matrices, "keys").normalized()

    plain = solve_ridge(stats, layers, 0, lam=5.0)
    whitened = solve_whitened(stats, layers, 0, metrics, N_KV_HEADS, HEAD_DIM, lam=5.0)

    # Reconstruct per-coordinate residuals from the cached moments.
    def rss(fit):
        xtx, xty, x_sum = stats.select(layers)
        xtx = xtx.double()
        xty = xty[:, 0].double()
        w = fit.weight.double()
        return (
            stats.yty_diag[0].double()
            - 2.0 * (w * xty).sum(0)
            + (w * (xtx @ w)).sum(0)
        )

    emphasized_idx = torch.tensor(emphasized)
    ignored_idx = torch.tensor(list(range(HEAD_DIM // 2, HEAD_DIM)))

    assert rss(whitened)[emphasized_idx].sum() < rss(plain)[emphasized_idx].sum(), (
        "whitened fit should be better on the emphasized coordinates"
    )
    assert rss(whitened)[ignored_idx].sum() > rss(plain)[ignored_idx].sum(), (
        "whitened fit should trade away accuracy on the downweighted ones"
    )


def test_key_metric_is_isotropic_for_fast_rotating_bands():
    """High-frequency bands rotate through many turns, washing out structure.

    Averaging over offsets must leave those 2x2 blocks isotropic, so the metric
    reduces to a per-band energy weight there.
    """
    head_dim, theta = 8, 100.0
    c = torch.zeros(1, 1, head_dim, head_dim, dtype=torch.float64)
    half = head_dim // 2
    # Strongly anisotropic input on the fastest band (index 0).
    c[0, 0, 0, 0] = 4.0
    c[0, 0, half, half] = 1.0
    c[0, 0, 0, half] = c[0, 0, half, 0] = 1.5

    metric = key_metric(c, head_dim, theta, max_offset=4096, n_offsets=4096).matrices

    top = metric[0, 0, 0, 0].item()
    bottom = metric[0, 0, half, half].item()
    off = metric[0, 0, 0, half].item()

    # Band energy is conserved: the average of the two coordinates is exactly
    # the average of the inputs.
    assert (top + bottom) / 2 == pytest.approx(2.5, rel=1e-3)

    # The 4:1 anisotropy of the input is very nearly flattened. It does not
    # vanish completely because offsets are sampled log-uniformly, which
    # oversamples short range and so leaves a little of the original structure
    # -- a fair reflection of where attention mass actually sits, not an error.
    input_ratio = 4.0
    output_ratio = max(top, bottom) / min(top, bottom)
    assert output_ratio < 1.1, f"fast band should be near-isotropic, got {output_ratio:.3f}"
    assert output_ratio < input_ratio / 3

    # Likewise the coupling between the band's two coordinates is suppressed by
    # more than an order of magnitude rather than annihilated.
    assert abs(off) < abs(1.5) / 10, f"off-diagonal barely suppressed: {off:.3f}"


def test_key_metric_preserves_structure_for_slow_bands():
    """The slowest bands barely rotate over realistic contexts.

    There R_delta is nearly the identity, so the metric must retain the query
    covariance's own anisotropy rather than flattening it. Handling both
    regimes is why the metric averages over offsets instead of assuming one.
    """
    head_dim, theta = 8, 1e6
    half = head_dim // 2
    c = torch.zeros(1, 1, head_dim, head_dim, dtype=torch.float64)
    slow = half - 1  # lowest frequency band
    c[0, 0, slow, slow] = 4.0
    c[0, 0, slow + half, slow + half] = 1.0

    freqs = rope_frequencies(head_dim, theta)
    assert freqs[slow] < 1e-2, "expected the last band to rotate very slowly"

    metric = key_metric(c, head_dim, theta, max_offset=64, n_offsets=256).matrices

    assert metric[0, 0, slow, slow].item() == pytest.approx(4.0, rel=1e-2)
    assert metric[0, 0, slow + half, slow + half].item() == pytest.approx(1.0, rel=1e-2)


def test_key_metric_is_positive_semidefinite():
    """Whatever the inputs, the metric must remain a valid metric."""
    torch.manual_seed(3)
    head_dim = 16
    a = torch.randn(2, 2, head_dim, head_dim, dtype=torch.float64)
    c = a @ a.transpose(-1, -2)  # PSD by construction

    metric = key_metric(c, head_dim, 1e6, max_offset=1024).matrices
    evals = torch.linalg.eigvalsh(metric)

    assert evals.min().item() > -1e-8, f"metric is not PSD: min eigenvalue {evals.min():.2e}"


@pytest.mark.slow
def test_value_metric_from_a_real_model():
    """The value metric must be computable and sane on real weights.

    Reads o_proj on whatever device the model sits on, which is where an MPS
    float64 restriction previously surfaced only at runtime.
    """
    import torch as t

    from kvxfer.geometry import load_geometry
    from kvxfer.metrics import value_metric
    from kvxfer.models import load_model

    model_id = "Qwen/Qwen3-0.6B"
    geom = load_geometry(model_id)
    model = load_model(model_id, dtype=t.float32)

    metrics = value_metric(model, geom)

    assert metrics.matrices.shape == (
        geom.n_layers, geom.n_kv_heads, geom.head_dim, geom.head_dim
    )
    assert metrics.matrices.dtype == t.float64
    evals = t.linalg.eigvalsh(metrics.matrices)
    assert evals.min().item() > -1e-8, "W_O' W_O must be PSD"
    assert evals.max().item() > 0, "metric should not be degenerate"

    # Normalizing must leave each head at unit mean diagonal.
    normalized = metrics.normalized()
    mean_diag = normalized.matrices.diagonal(dim1=-2, dim2=-1).mean(-1)
    assert t.allclose(mean_diag, t.ones_like(mean_diag), atol=1e-6)


def test_metric_r2_matches_direct_computation():
    """Metric-weighted R² from moments must equal the value from samples.

    The expansion involves the prediction's own second moment and its cross
    moment with the target, including bias terms in both -- easy to get subtly
    wrong, and wrong in a way that would quietly distort the central claim this
    quantity exists to test.
    """
    from kvxfer.solvers.whitened import held_out_metric_r2

    torch.manual_seed(11)
    dim_in = N_SRC_LAYERS * KV_DIM
    w_true = torch.randn(dim_in, KV_DIM, dtype=torch.float64) / dim_in**0.5
    b_true = torch.randn(KV_DIM, dtype=torch.float64)

    fit_stats = _stats(noise=0.4, seed=11)
    fit = solve_ridge(fit_stats, tuple(range(N_SRC_LAYERS)), 0, lam=1e-2)

    # Independent validation split, kept as samples for direct scoring.
    x_val = torch.randn(2500, dim_in, dtype=torch.float64)
    y_val = (
        x_val @ w_true + b_true + 0.4 * torch.randn(2500, KV_DIM, dtype=torch.float64)
    )
    acc = GramAccumulator(_geom(N_SRC_LAYERS), _geom(N_TGT_LAYERS), kind="keys")
    acc.update(x_val, torch.stack([y_val, torch.randn_like(y_val)], dim=1))
    val_stats = acc.finalize()

    # A deliberately anisotropic metric, so the weighting actually bites.
    torch.manual_seed(12)
    raw = torch.randn(N_TGT_LAYERS, N_KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    metrics = HeadMetrics(raw @ raw.transpose(-1, -2), "keys")

    from_moments = held_out_metric_r2(val_stats, fit, metrics, N_KV_HEADS, HEAD_DIM)

    pred = x_val @ fit.weight.double() + fit.bias.double()
    residual = pred - y_val
    centered = y_val - y_val.mean(0)

    scores = []
    for head in range(N_KV_HEADS):
        lo, hi = head * HEAD_DIM, (head + 1) * HEAD_DIM
        m = metrics.matrices[0, head]
        rss = torch.einsum("ni,ij,nj->", residual[:, lo:hi], m, residual[:, lo:hi])
        tss = torch.einsum("ni,ij,nj->", centered[:, lo:hi], m, centered[:, lo:hi])
        scores.append(1.0 - rss / tss)
    direct = float(torch.stack(scores).mean())

    # Agreement is bounded by the accumulator's float32 storage, not by the
    # float64 solve: the moments themselves carry ~1e-7 relative error, so a
    # tighter tolerance here would be testing the storage precision rather
    # than the correctness of the expansion.
    assert from_moments == pytest.approx(direct, rel=1e-4, abs=1e-6), (
        f"moment-based metric R2 {from_moments:.8f} != direct {direct:.8f}"
    )


def test_whitened_wins_on_its_own_objective():
    """The whitened fit must beat plain ridge under the metric it optimizes.

    Ridge is optimal for isotropic error by construction, so it must win on
    plain R². The whitened solve must win on the metric-weighted score. If
    either ordering fails, the objective is not doing what it claims.
    """
    from kvxfer.solvers.whitened import held_out_metric_r2

    # Fit and validation must come from the same generating process; _stats
    # redraws the true weights per seed, so both splits use one seed and differ
    # only through the noise realization drawn within it.
    fit_stats = _stats(noise=0.6, seed=13, n_tokens=8000)
    val_stats = _stats(noise=0.6, seed=13, n_tokens=8000)
    layers = tuple(range(N_SRC_LAYERS))

    raw = torch.randn(N_TGT_LAYERS, N_KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    metrics = HeadMetrics(raw @ raw.transpose(-1, -2), "keys").normalized()

    plain = solve_ridge(fit_stats, layers, 0, lam=1.0)
    whitened = solve_whitened(
        fit_stats, layers, 0, metrics, N_KV_HEADS, HEAD_DIM, lam=1.0
    )

    from kvxfer.solvers.ridge import held_out_r2

    plain_iso = float(held_out_r2(val_stats, plain).mean())
    whitened_iso = float(held_out_r2(val_stats, whitened).mean())
    plain_metric = held_out_metric_r2(val_stats, plain, metrics, N_KV_HEADS, HEAD_DIM)
    whitened_metric = held_out_metric_r2(
        val_stats, whitened, metrics, N_KV_HEADS, HEAD_DIM
    )

    assert plain_iso > whitened_iso, (
        f"ridge should win on isotropic R2: {plain_iso:.5f} vs {whitened_iso:.5f}"
    )
    assert whitened_metric > plain_metric, (
        f"whitened should win on the metric it optimizes: "
        f"{whitened_metric:.5f} vs {plain_metric:.5f}"
    )


def _anisotropic_metrics(seed: int = 3) -> HeadMetrics:
    """A strongly anisotropic metric, so alpha has something to act on."""
    torch.manual_seed(seed)
    a = torch.randn(N_TGT_LAYERS, N_KV_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    return HeadMetrics(a @ a.transpose(-1, -2) / HEAD_DIM, "keys").normalized()


def test_alpha_zero_reduces_to_plain_ridge_for_any_metric():
    """At alpha=0 the metric must drop out entirely, however anisotropic it is.

    This is a stronger control than the identity-metric test: it pins that the
    metric enters *only* through the penalty exponent, so a measured difference
    between the objectives cannot come from the rotation machinery.
    """
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    metrics = _anisotropic_metrics()

    plain = solve_ridge(stats, layers, 0, lam=1e-2)
    flat = solve_whitened(
        stats, layers, 0, metrics, N_KV_HEADS, HEAD_DIM, lam=1e-2, alpha=0.0
    )

    assert torch.allclose(plain.weight, flat.weight, atol=1e-6), (
        f"max weight difference {(plain.weight - flat.weight).abs().max():.2e}"
    )
    assert torch.allclose(plain.bias, flat.bias, atol=1e-6)


def test_alpha_interpolates_between_the_two_objectives():
    """Increasing alpha must move the fit monotonically away from ridge.

    Anything else would mean alpha is not the knob it claims to be, and tuning
    it out of sample would be tuning noise.
    """
    stats = _stats()
    layers = tuple(range(N_SRC_LAYERS))
    metrics = _anisotropic_metrics()

    plain = solve_ridge(stats, layers, 0, lam=1e-2)
    distances = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        fit = solve_whitened(
            stats, layers, 0, metrics, N_KV_HEADS, HEAD_DIM, lam=1e-2, alpha=alpha
        )
        distances.append(float((fit.weight - plain.weight).norm()))

    assert distances[0] == pytest.approx(0.0, abs=1e-6)
    assert all(
        b > a for a, b in zip(distances, distances[1:])
    ), f"not monotone in alpha: {distances}"


def test_query_capture_handles_families_without_per_head_norm():
    """Queries must be collectable from q_proj, not only from q_norm.

    Qwen3 normalizes queries per head and so exposes them already shaped
    (batch, seq, heads, dim). Qwen2, Mistral and Llama have no such module, and
    hooking q_proj yields the same vectors flattened to (batch, seq,
    heads * dim). Assuming the first shape sent a Qwen2.5 run to an
    AttributeError partway into a paid job.
    """
    from kvxfer.queries import _QueryMomentHook

    geom = _geom(N_TGT_LAYERS)
    batch, seq = 2, 4
    torch.manual_seed(0)
    flat = torch.randn(batch, seq, geom.n_q_heads * geom.head_dim)
    shaped = flat.view(batch, seq, geom.n_q_heads, geom.head_dim)

    from_proj = _QueryMomentHook(geom, torch.device("cpu"))
    from_proj.make(0)(None, None, flat)
    from_norm = _QueryMomentHook(geom, torch.device("cpu"))
    from_norm.make(0)(None, None, shaped)

    assert from_proj.n_tokens == from_norm.n_tokens
    assert torch.allclose(from_proj.moments, from_norm.moments, atol=1e-6), (
        "the two capture points disagree; the metric would depend on architecture"
    )
