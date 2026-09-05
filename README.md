# kvxfer — Attention-Aligned KV Cache Transfer

Reuse a small model's prefill by *mapping* its KV cache into a larger
same-family model's format, instead of re-prefilling from scratch.

This is a framework and study building on **"Cross-Model KV Cache Transfer in
LLM Families: A Closed-Form Linear Mapping for Prefill Reuse"**
([arXiv:2608.03893](https://arxiv.org/abs/2608.03893), Heo, Shafipour, Zhao
et al., NVIDIA), which fits per-head ridge regressions from source KV to target
KV in RoPE-stripped content space.

## The idea

That paper reports a striking diagnostic: **attention-output cosine predicts
downstream retention (r = +0.57), while calibration R² *anti*-correlates
(r = −0.20)**. Fitting the KV cache better in a least-squares sense makes
downstream accuracy *worse*.

That is a symptom of optimizing the wrong objective. Plain MSE treats all 128
dimensions of an attention head as equally important, but they are not: an
error `ε` in a key reaches the model only through the logit `qᵀε`, so its true
cost is `εᵀ E[qqᵀ] ε` — a Mahalanobis metric, not the identity. An error in a
value reaches the residual stream only through `W_O`, scaled by how much
attention that token actually receives.

`kvxfer` replaces the isotropic objective with an **attention-aligned** one that
stays closed-form, and adds a trained residual mapper on top of it.

## Status

Under active development. See `docs/` for the running experiment log.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev,plot]"
```

## Tests

```bash
uv run pytest              # fast correctness gates
uv run pytest -m slow      # gates that load real model weights
```
