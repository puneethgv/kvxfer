# kvxfer — cross-model KV cache transfer, measured honestly

Reuse a small model's prefill by *mapping* its KV cache into a larger
same-family model's format, instead of re-prefilling from scratch.

This replicates and extends **"Cross-Model KV Cache Transfer in LLM Families: A
Closed-Form Linear Mapping for Prefill Reuse"**
([arXiv:2608.03893](https://arxiv.org/abs/2608.03893), Heo, Shafipour, Zhao et
al., NVIDIA), which fits per-head ridge regressions from source KV to target KV
in RoPE-stripped content space.

**The replication holds.** On the layer-count-matched pair the mapped cache
retains **94.6%** of ARC-Easy accuracy, inside the reference work's reported
73–98% band.

What this repository adds is three things that band cannot show you: a baseline
that reveals when transfer stops being worth doing, the first wall-clock numbers
for the method against a production engine, and five attempts to improve the
mapping — four of which failed, in ways that say something specific about the
problem.

---

## Findings

### 1. Retention against the target hides when the method stops paying

Retention is measured against the target model. It never asks the question a
practitioner actually faces: *is this better than just running the small model?*

| pair | ARC-Easy retention | mapped vs. **source model** |
|---|---|---|
| 0.6B → 1.7B | 94.6% | **+0.1000** (p = 0.001) — clearly worth it |
| 1.7B → 4B | 87.1% | −0.0067 (p = 0.90) — no benefit |
| 0.6B → 4B | 72.5% | +0.0200 (p = 0.55) — no benefit |

Retention still reads as a healthy 72–87% across exactly the region where the
mapped cache stops beating the model you already ran. The metric cannot see the
crossing point; the source baseline can. It is one line of code and it is
missing from the reference work.

### 2. Mapping is ~4× cheaper than prefilling — measured against vLLM

Qwen3-1.7B → 4B on an L4, against vLLM with prefix caching **off** (with it on,
vLLM "prefilled" 8192 tokens in 74.5 ms, which is below the 544 ms floor the
card's peak throughput allows — it was timing cache hits):

| tokens | vLLM prefill | map + inject | **warm** | cold |
|---|---|---|---|---|
| 512 | 106.1 ms | 23.3 ms | **4.56×** | 1.48× |
| 1024 | 175.1 ms | 44.9 ms | **3.90×** | 1.38× |
| 2048 | 332.9 ms | 92.5 ms | **3.60×** | 1.30× |
| 4096 | 723.9 ms | 204.0 ms | **3.55×** | 1.26× |
| 8192 | 1653.0 ms | 418.6 ms | **3.95×** | 1.29× |

**warm** assumes the source model already ran — the escalation case the method
is for, where its prefill is sunk cost. **cold** pays for it, and is much
weaker. Reporting only the warm figure would inflate the method the same way
retention does.

The map reaches 15 TFLOPS of the card's ~121, so roughly 8× of implementation
headroom remains against a 13.3× FLOP ceiling.

### 3. A trained residual is the only thing that improved on plain ridge

Four closed-form variants failed. The fifth attempt — a small network trained on
attention-output error rather than reconstruction error — is the first to help:

| condition | perplexity | ARC-Easy | ARC-Challenge |
|---|---|---|---|
| target (ceiling) | 13.559 | 0.8500 | 0.4967 |
| **source model** | 16.479 | **0.7467** | **0.3800** |
| **+ trained residual** | **14.108** | 0.7333 | **0.3533** |
| ridge (reference method) | 14.708 | 0.7400 | 0.3233 |
| floor (zeroed cache) | 19.635 | 0.4267 | 0.1900 |

- **Perplexity: −0.04167 ± 0.00558 nats/token, t = −7.46, 52/64 documents
  improved.** It closes **51%** of the gap ridge leaves to the target.
- **ARC-Challenge: +0.0300 (p = 0.078)**, retention 65.1% → 71.1%.
- **ARC-Easy: −0.0067 (p = 0.79)** — nothing.

So it is a decisive win on generation quality, a marginal one on the harder
task, and **still short of simply running the source model** on both tasks.

### 4. The metric decides the conclusion, repeatedly

This is the reference work's own diagnostic (calibration R² *anti*-correlates
with retention, r = −0.20) recurring at every level:

- Its **training** objective improved 3.0% while downstream perplexity improved
  **51%** — the training curve looked like failure for two hours.
- The same residual, same caches, same run: **51% better perplexity, 0% better
  ARC-Easy.**
- Rank-128 maps hold **R² = 0.54** and score *worse than a zeroed cache*.

Any single number here supports a different conclusion about the same artifact.

---

## What failed, and what that says

| attempt | result |
|---|---|
| Attention-aligned penalty (`λ/Λⱼ`) | Null on 3 pairs; **α tunes to 0 out of sample**, i.e. the tuned solver *is* ridge |
| The α family across λ | Monotonically worse; at the selected λ, metric-R² is flat in α while isotropic R² falls |
| Metric-weighted low-rank | Worse at **every** rank (t = +22.7 at rank 512, 0/64 documents improved) |
| Rank truncation | Graceful to 512, then collapses: 24.30 ppl at 256, 44.28 at 128, past the 27.33 floor |
| **Trained residual** | **The only one that helps** |

The first four share a cause. In a closed form the attention metric can only
enter *through the penalty*, and reallocating a penalty is second-order against
a design Gram conditioned at 4×10¹¹. Sweeping its influence honestly drives it
to zero. The trained residual is the only variant that optimizes the functional
objective **directly**, and it is the only one that works — which is evidence
about the constraint, not about attention alignment being wrong.

What the residual's ceiling suggests: the target's cache appears predictable
from the source's largely to the extent that it is a *linear* function of it.
Ridge recovers that, and 94.4M trained parameters recover about half of what is
left in perplexity terms and little of it in accuracy terms.

---

## Reproducing

```bash
# 1. calibrate — one streaming pass, keeps only sufficient statistics
python scripts/calibrate.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B

# 2. choose the penalty out of sample, on each solver's own objective
python scripts/sweep_lambda.py --artifacts artifacts/<pair>/web --target Qwen/Qwen3-4B

# 3. fit maps once and store them
python scripts/fit_maps.py --artifacts artifacts/<pair>/web --out maps/<pair>

# 4. evaluate; --maps skips refitting
python scripts/fit_and_eval.py --artifacts artifacts/<pair>/web --maps maps/<pair>

# 5. optional: train the residual on the attention-output objective
python scripts/train_residual.py --artifacts artifacts/<pair>/web \
    --maps maps/<pair> --out maps/<pair>__residual

# regenerate every table from committed artifacts
python scripts/make_table.py
```

Pairs too large for a laptop run on Modal. **Always `--detach`** — without it
the app is tied to the local client, and a local out-of-memory kill will stop a
GPU run that has nothing to do with it:

```bash
modal run --detach modal_app.py --source Qwen/Qwen3-1.7B --target Qwen/Qwen3-4B
modal run --detach modal_app.py::latency --artifacts <pair>/web
modal run --detach modal_app.py::vllm_prefill --model Qwen/Qwen3-4B
```

## Protocol

Things that changed the numbers, and are therefore worth stating:

- **Out-of-sample selection.** `k`, `λ` and `α` are chosen on a held-out
  calibration split, never on the evaluation set. Held-out R² is computed from
  *moments alone*, so a validation split costs one accumulator pass and no
  stored activations.
- **Paired tests.** Conditions are scored on identical items, so per-condition
  error bars overstate the uncertainty of the difference between them. Exact
  McNemar for accuracy, per-document paired t for perplexity.
- **A source-model baseline.** See finding 1.
- **Prefix-conditioned perplexity as the primary metric**, one measurement per
  token rather than per item. The multiple-choice tasks leave so little headroom
  on easy pairs that per-item noise swamps the effect.
- **bfloat16 throughout**, matching how the statistics were harvested and how
  these models are served. Logits are upcast to float32 before `log_softmax`.
- **Correctness gates as tests.** Identity injection must reproduce the target's
  own scores exactly; RoPE must round-trip; an oracle mapper must give 100%
  retention. 78 tests.

## Requirements

The mapping needs **matched KV geometry** — identical `n_kv_heads` *and*
`head_dim`. The whole Qwen3 dense family qualifies (8 KV heads × 128). Note
that Qwen2.5 does *not* match across its own sizes: 0.5B is `head_dim=64`
against 1.5B's 128, and 7B has 4 KV heads against 1.5B's 2, leaving
**1.5B → 3B** as the only matched pair there.

Map cost scales with the *square* of `kv_dim`, so narrow-GQA families are far
cheaper to serve: the same map is 302M parameters for Qwen3 (8 KV heads) and
18.9M for Qwen2.5 (2 KV heads).

## Layout

```
kvxfer/
  stats.py       streaming sufficient statistics; one pass serves every (k, λ, subset)
  solvers/
    ridge.py     the reference method
    whitened.py  attention-aligned closed form, with the α family
    lowrank.py   metric-weighted reduced-rank regression
    neural.py    trained residual on the attention-output objective
  eval/
    ppl.py       prefix-conditioned perplexity (primary)
    retention.py multiple-choice scoring with paired tests
    latency.py   map-and-inject vs. re-prefill, warm and cold
  planning.py    measured-memory decisions, not guessed ones
```

`stats.py` is the load-bearing design: one Gram over *all* candidate source
layers means every `(k, subset, λ)` choice is a submatrix solve, so one GPU pass
buys an unlimited number of free ablations.
