**All test pairs**

| | Pairs | Agrees with human majority (95% CI) | Clear-cut pairs (≥8–2) | Pearson r vs vote share | Order consistency |
|---|---:|---|---:|---:|---:|
| Muse Spark 1.3 | 607 | **82.6%** (79.2%–85.6%) | 87.9% | 0.75 | 87.6% |
| Single human vs. other 9 (ceiling) | 607 | 77.4% (75.7%–79.1%) | – | – | – |

**Head-to-head on the same 150 pairs**

| | Pairs | Agrees with human majority (95% CI) | Clear-cut pairs (≥8–2) | Pearson r vs vote share | Order consistency |
|---|---:|---|---:|---:|---:|
| Muse Spark 1.3 | 150 | **81.8%** (75.4%–88.2%) | 89.9% | 0.73 | 87.8% |
| Muse Wonder 1.0 | 150 | **85.0%** (78.7%–90.7%) | 92.7% | 0.80 | 92.5% |
| Single human vs. other 9 (ceiling) | 150 | 76.1% (72.6%–79.3%) | – | – | – |

**Model ranking on `cup_stacking`** (Bradley-Terry Elo from human votes vs. from the judge's verdicts)

| Rank | Humans | Elo | Muse Spark judge | Elo |
|---:|---|---:|---|---:|
| 1 | π0.5-DROID | 1058 | MolmoAct 2 | 1048 |
| 2 | OpenVLA-OFT | 1051 | OpenVLA-OFT | 1044 |
| 3 | Cosmos Policy | 1039 | Cosmos Policy | 1043 |
| 4 | MolmoAct 2 | 1033 | π0.5-DROID | 1030 |
| 5 | X-VLA | 820 | X-VLA | 836 |

Judge Elo is within **14 points** of human Elo on average (max 28). Both put X-VLA clearly last (~213 Elo below the rest). The other four sit within 25 Elo of each other even on human votes, so their order is within noise for humans and judge alike.

Muse Spark agreement by layout: in-distribution: 83.4% (420 pairs); out-of-distribution: 80.6% (187 pairs).
