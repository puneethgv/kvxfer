# kvxfer — Attention-Aligned KV Cache Transfer

Reuse a small model's prefill by *mapping* its KV cache into a larger
same-family model's format, instead of re-prefilling from scratch.

This builds on **"Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse"** ([arXiv:2608.03893](https://arxiv.org/abs/2608.03893),
Heo, Shafipour, Zhao et al., NVIDIA), which fits per-head ridge regressions from
source KV to target KV in RoPE-stripped content space.

## The idea

That paper reports a striking diagnostic: **attention-output cosine predicts
downstream retention (r = +0.57), while calibration R² *anti*-correlates
(r = −0.20)**. Fitting the KV cache better in a least-squares sense makes
downstream accuracy *worse*.

That is a symptom of optimizing the wrong objective. Plain MSE treats all 128
dimensions of an attention head as equally important, but they are not. A key
error `ε` reaches the model only through the attention logit `qᵀε`, so its true
cost is `εᵀ E[qqᵀ] ε` — a Mahalanobis metric, not the identity. A value error
reaches the residual stream only through `W_O`, scaled by how much attention
that token actually receives.

`kvxfer` replaces the isotropic objective with an **attention-aligned** one
that stays closed-form: diagonalizing the metric decouples the fit into one
independent ridge per output coordinate with penalty `λ/Λⱼ`, so a single
eigendecomposition of the design Gram serves all of them.

## Design

**Calibration is one streaming pass that stores no activations.** Ridge needs
only `X'X` and `X'Y`, so the Gram is accumulated over *all* candidate source
layers at once and every later choice of subset is a submatrix. Layer
selection, penalty sweeps, and both solvers then run off cached statistics with
no further GPU work — which is what makes a study of this shape affordable.

**Model family.** The entire Qwen3 dense family is matched-KV — 8 KV heads ×
128 dim, `rope_theta` 1e6, shared tokenizer at every size — giving more transfer
pairs within one family than the reference work used across three.

```
kvxfer/
  geometry.py   stats.py     harvest.py    metrics.py    queries.py
  rope.py       cache.py     mappers.py    data.py
  solvers/      ridge.py (baseline)  whitened.py (attention-aligned)
  eval/         scoring.py   retention.py  tasks.py
```

## Correctness gates

Four tests decide whether any downstream number can be trusted. They use
mappers that do no work, so they test the harness rather than the method:

1. **Identity injection** — injecting a model's own cache reproduces its
   standalone scores (< 1e-2 nats).
2. **RoPE round trip** — strip/restore is exact, and `apply_rope` matches
   transformers' reference implementation.
3. **Oracle mapper** — ground-truth KV fed deliberately corrupted input is exact.
4. **Position relocation** — a cache stripped at one set of positions and
   re-rotated at another matches a genuine prefill there (offsets to 4096).

Plus a floor check: zeroing the cache must move the score, or the harness is
not really using it.

## Usage

```bash
uv venv --python 3.12
uv pip install -e ".[dev,plot]"

# One calibration pass; writes sufficient statistics.
python scripts/calibrate.py --source Qwen/Qwen3-0.6B --target Qwen/Qwen3-1.7B

# Fit both solvers from cached statistics and measure retention. No GPU
# calibration is repeated here.
python scripts/fit_and_eval.py \
    --artifacts artifacts/qwen3-0.6b__to__qwen3-1.7b/web \
    --tasks arc_easy,hellaswag

python scripts/make_table.py          # regenerate tables from results/
```

For pairs too large for local memory, `modal_app.py` runs calibration on rented
GPUs. Weights live on a Volume so a multi-gigabyte download happens once, and
solving deliberately has no remote entrypoint — it is CPU linear algebra over
downloaded statistics and should never hold a GPU.

## Tests

```bash
uv run pytest              # everything
uv run pytest -m slow      # gates that load real model weights
```
