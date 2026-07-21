## Bang tong hop grid (mean±std, 3 seed, FP32) — NELL

| Phương pháp | θ | Valid MRR (chọn) | Test MRR (báo cáo) | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Hybrid+Rerank | 1% | 0.5433 | 0.5430 ± 0.0054 | 14.79 | 282 |
| Hybrid+Rerank | 5% | 0.5586 | 0.5458 ± 0.0036 | 21.03 | 1046 |
| Hybrid+Rerank | 10% | 0.5688 | 0.5455 ± 0.0033 | 30.14 | 1740 |
| Hybrid+Rerank | 20% | 0.5632 | 0.5427 ± 0.0045 | 47.04 | 2590 |
| PPR-only | 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 | 395 |
| PPR-only | 5% | 0.5012 | 0.5369 ± 0.0042 | 16.19 | 1715 |
| PPR-only | 10% | 0.5003 | 0.5354 ± 0.0036 | 27.12 | 2547 |
| PPR-only | 20% | 0.4959 | 0.5297 ± 0.0050 | 46.37 | 3573 |
| PIVOT-Rerank | 1% | 0.5357 | 0.5414 ± 0.0044 | 15.17 | 395 |
| PIVOT-Rerank | 5% | 0.5655 | 0.5468 ± 0.0042 | 27.77 | 1715 |
| PIVOT-Rerank | 10% | 0.5684 | 0.5448 ± 0.0038 | 41.78 | 2547 |
| PIVOT-Rerank | 20% | 0.5590 | 0.5384 ± 0.0060 | 70.96 | 3573 |

## Frontier co so (PPR-only) — deliverable T7-8 thuan

| θ | Valid MRR | Test MRR | Latency/q (ms) |
|:---:|:---:|:---:|:---:|
| 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 |
| 5% | 0.5012 | 0.5369 ± 0.0042 | 16.19 |

## Frontier day du — deliverable tich hop T7-9

| Phương pháp | θ | Valid MRR | Test MRR | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PPR-only | 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 | 395 |
| Hybrid+Rerank | 1% | 0.5433 | 0.5430 ± 0.0054 | 14.79 | 282 |
| Hybrid+Rerank | 5% | 0.5586 | 0.5458 ± 0.0036 | 21.03 | 1046 |
| PIVOT-Rerank | 5% | 0.5655 | 0.5468 ± 0.0042 | 27.77 | 1715 |
| Hybrid+Rerank | 10% | 0.5688 | 0.5455 ± 0.0033 | 30.14 | 1740 |

## Demo BudgetController (chon theo Valid, bao cao Test)

- **latency_constrained**: Hybrid+Rerank @ θ=10% → Valid 0.5688, Test 0.5455 ± 0.0033, 30.14 ms/q, 1740 MB
- **accuracy_constrained**: Hybrid+Rerank @ θ=1% → Valid 0.5433, Test 0.5430 ± 0.0054, 14.79 ms/q, 282 MB