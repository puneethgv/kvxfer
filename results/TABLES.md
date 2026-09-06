# Results

### Qwen3-0.6B to Qwen3-1.7B

Calibrated on 131,072 tokens (32,768 held out), k=4 source layers per target layer.

**arc_easy** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.747 ± 0.025 | — | — |
| floor | 0.353 ± 0.028 | 47.3% | — |
| ridge | 0.707 ± 0.026 | 94.6% | 89.8% |
| whitened | 0.710 ± 0.026 | 95.1% | 90.7% |
| source | 0.607 ± 0.028 | 81.2% | 64.4% |

Paired ridge vs whitened: +0.0033 accuracy, exact McNemar p=1.000.

**arc_challenge** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.377 ± 0.028 | — | — |
| floor | 0.167 ± 0.022 | 44.2% | — |
| ridge | 0.317 ± 0.027 | 84.1% | 71.4% |
| whitened | 0.310 ± 0.027 | 82.3% | 68.3% |
| source | 0.280 ± 0.026 | 74.3% | 54.0% |

Paired ridge vs whitened: -0.0067 accuracy, exact McNemar p=0.688.

**Prefix-conditioned perplexity** (primary metric)

| condition | perplexity | mean NLL | stored params |
| --- | --- | --- | --- |
| target | 16.4696 | 2.80152 ± 0.04968 | — |
| floor | 27.3281 | 3.30792 ± 0.14187 | — |
| ridge | 17.4520 | 2.85946 ± 0.04944 | 234.9M |
| whitened | 17.4437 | 2.85898 ± 0.04944 | 234.9M |
| source | 21.2269 | 3.05527 ± 0.05107 | — |

Paired by document, so the comparison is not read off overlapping per-condition error bars:

| comparison | mean NLL difference | t | documents improved |
| --- | --- | --- | --- |
| ridge vs whitened | -0.00048 ± 0.00053 | -0.91 | 34/64 |

### Held-out fit quality vs. retention

| variant | mean held-out R² (keys) | mean held-out R² (values) |
| --- | --- | --- |
| ridge | 0.6916 | 0.5964 |
| whitened | 0.6832 | 0.5964 |

Read this against the perplexity table rather than on its own. The reference work reports calibration R² anti-correlating with retention (r = -0.20), so a variant leading here is not thereby the better mapper -- that is the claim these runs are set up to check.
