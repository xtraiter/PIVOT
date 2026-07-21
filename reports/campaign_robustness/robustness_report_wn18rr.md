## Bảng Degradation — Robustness Suite WN18RR (mean±std, 3 seed, FP32, θ=10%)

| Config | Phương pháp | Test MRR | Δ vs clean | Retention % |
|:---:|:---:|:---:|:---:|:---:|
| clean | PPR-only | 0.5638 ± 0.0017 | +0.0000 | 100.0% |
| clean | PIVOT-Rerank | 0.5685 ± 0.0021 | +0.0000 | 100.0% |
| clean | Hybrid+Rerank | 0.5684 ± 0.0015 | +0.0000 | 100.0% |
| del05 | PPR-only | 0.5113 ± 0.0026 | -0.0525 | 90.7% |
| del05 | PIVOT-Rerank | 0.5147 ± 0.0029 | -0.0538 | 90.5% |
| del05 | Hybrid+Rerank | 0.5152 ± 0.0024 | -0.0532 | 90.6% |
| del10 | PPR-only | 0.4733 ± 0.0031 | -0.0905 | 83.9% |
| del10 | PIVOT-Rerank | 0.4769 ± 0.0034 | -0.0916 | 83.9% |
| del10 | Hybrid+Rerank | 0.4766 ± 0.0025 | -0.0918 | 83.9% |
| del20 | PPR-only | 0.4159 ± 0.0012 | -0.1479 | 73.8% |
| del20 | PIVOT-Rerank | 0.4202 ± 0.0017 | -0.1483 | 73.9% |
| del20 | Hybrid+Rerank | 0.4199 ± 0.0010 | -0.1485 | 73.9% |
| reldel | PPR-only | 0.5616 ± 0.0014 | -0.0022 | 99.6% |
| reldel | PIVOT-Rerank | 0.5663 ± 0.0021 | -0.0022 | 99.6% |
| reldel | Hybrid+Rerank | 0.5663 ± 0.0013 | -0.0021 | 99.6% |

## Khoảng cách Rerank − PPR theo mức nhiễu

| Config | Δ(Rerank − PPR) | Nhận xét |
|:---:|:---:|:---|
| clean | +0.0047 | Gap giu vung so voi clean (+0.0047) |
| del05 | +0.0034 | Gap thu hep so voi clean (+0.0047) |
| del10 | +0.0036 | Gap thu hep so voi clean (+0.0047) |
| del20 | +0.0043 | Gap giu vung so voi clean (+0.0047) |
| reldel | +0.0047 | Gap giu vung so voi clean (+0.0047) |
