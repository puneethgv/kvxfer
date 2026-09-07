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

### Qwen3-0.6B to Qwen3-4B

Calibrated on 131,072 tokens (32,768 held out), k=4 source layers per target layer.

**arc_easy** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.850 ± 0.021 | — | — |
| floor | 0.427 ± 0.029 | 50.2% | — |
| ridge | 0.617 ± 0.028 | 72.5% | 44.9% |
| whitened | 0.630 ± 0.028 | 74.1% | 48.0% |
| source | 0.597 ± 0.028 | 70.2% | 40.2% |

Paired ridge vs whitened: +0.0133 accuracy, exact McNemar p=0.289.

**arc_challenge** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.497 ± 0.029 | — | — |
| floor | 0.190 ± 0.023 | 38.3% | — |
| ridge | 0.263 ± 0.025 | 53.0% | 23.9% |
| whitened | 0.257 ± 0.025 | 51.7% | 21.7% |
| source | 0.287 ± 0.026 | 57.7% | 31.5% |

Paired ridge vs whitened: -0.0067 accuracy, exact McNemar p=0.625.

**Prefix-conditioned perplexity** (primary metric)

| condition | perplexity | mean NLL | stored params |
| --- | --- | --- | --- |
| target | 13.5588 | 2.60704 ± 0.04740 | — |
| floor | 19.6354 | 2.97733 ± 0.06367 | — |
| ridge | 14.7840 | 2.69354 ± 0.04624 | 302.1M |
| whitened | 14.8067 | 2.69508 ± 0.04635 | 302.1M |
| source | 21.2278 | 3.05531 ± 0.05109 | — |

Paired by document, so the comparison is not read off overlapping per-condition error bars:

| comparison | mean NLL difference | t | documents improved |
| --- | --- | --- | --- |
| ridge vs whitened | +0.00154 ± 0.00056 | +2.76 | 24/64 |

### Held-out fit quality vs. retention

| variant | mean held-out R² (keys) | mean held-out R² (values) |
| --- | --- | --- |
| ridge | 0.6515 | 0.5365 |
| whitened | 0.6176 | 0.5364 |

Read this against the perplexity table rather than on its own. The reference work reports calibration R² anti-correlating with retention (r = -0.20), so a variant leading here is not thereby the better mapper -- that is the claim these runs are set up to check.

### Qwen3-1.7B to Qwen3-4B

Calibrated on 131,072 tokens (32,768 held out), k=4 source layers per target layer.

**arc_easy** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.850 ± 0.021 | — | — |
| floor | 0.427 ± 0.029 | 50.2% | — |
| ridge | 0.740 ± 0.025 | 87.1% | 74.0% |
| whitened | 0.733 ± 0.026 | 86.3% | 72.4% |
| source | 0.747 ± 0.025 | 87.8% | 75.6% |

Paired ridge vs whitened: -0.0067 accuracy, exact McNemar p=0.625.

**arc_challenge** (300 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.497 ± 0.029 | — | — |
| floor | 0.190 ± 0.023 | 38.3% | — |
| ridge | 0.323 ± 0.027 | 65.1% | 43.5% |
| whitened | 0.317 ± 0.027 | 63.8% | 41.3% |
| source | 0.380 ± 0.028 | 76.5% | 62.0% |

Paired ridge vs whitened: -0.0067 accuracy, exact McNemar p=0.625.

**Prefix-conditioned perplexity** (primary metric)

| condition | perplexity | mean NLL | stored params |
| --- | --- | --- | --- |
| target | 13.5588 | 2.60704 ± 0.04740 | — |
| floor | 19.6354 | 2.97733 ± 0.06367 | — |
| ridge | 14.7084 | 2.68842 ± 0.04657 | 302.1M |
| whitened | 14.7231 | 2.68942 ± 0.04653 | 302.1M |
| source | 16.4793 | 2.80210 ± 0.04976 | — |

Paired by document, so the comparison is not read off overlapping per-condition error bars:

| comparison | mean NLL difference | t | documents improved |
| --- | --- | --- | --- |
| ridge vs whitened | +0.00100 ± 0.00055 | +1.83 | 24/64 |

### Held-out fit quality vs. retention

| variant | mean held-out R² (keys) | mean held-out R² (values) |
| --- | --- | --- |
| ridge | 0.6821 | 0.5725 |
| whitened | 0.6549 | 0.5723 |

Read this against the perplexity table rather than on its own. The reference work reports calibration R² anti-correlating with retention (r = -0.20), so a variant leading here is not thereby the better mapper -- that is the claim these runs are set up to check.
