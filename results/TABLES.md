# Results

### Qwen3-0.6B to Qwen3-1.7B

Calibrated on 131,072 tokens (32,768 held out), k=4 source layers per target layer.

**arc_easy** (250 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.732 ± 0.028 | — | — |
| floor | 0.348 ± 0.030 | 47.5% | — |
| ridge | 0.692 ± 0.029 | 94.5% | 89.6% |
| whitened | 0.700 ± 0.029 | 95.6% | 91.7% |

**hellaswag** (250 items)

| condition | accuracy | retention | floor-normalized |
| --- | --- | --- | --- |
| target | 0.464 ± 0.032 | — | — |
| floor | 0.384 ± 0.031 | 82.8% | — |
| ridge | 0.440 ± 0.031 | 94.8% | 70.0% |
| whitened | 0.444 ± 0.031 | 95.7% | 75.0% |

### Held-out fit quality vs. retention

| variant | mean held-out R² (keys) | mean held-out R² (values) |
| --- | --- | --- |
| ridge | 0.6893 | 0.5946 |
| whitened | 0.5864 | 0.5945 |
