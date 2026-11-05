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

**Every task.** Muse Spark on a random sample of pairs from the 7 training tasks (these tasks were never used to tune the prompt either).

| Task | Pairs | Judge agrees with majority | Single human vs. other 9 |
|---|---:|---:|---:|
| `candy_sorting` | 53 | 93.3% | 67.7% |
| `cup_storing` | 46 | 87.5% | 77.4% |
| `pegboard` | 58 | 69.4% | 68.8% |
| `pipette` | 58 | 76.9% | 72.1% |
| `popcorn_microwave` | 43 | 81.1% | 64.7% |
| `test_tubes` | 42 | 70.3% | 68.1% |
| `toy_storing` | 49 | 87.8% | 69.6% |
| `cup_stacking (test)` | 607 | 82.6% | 77.4% |
| **All 8 tasks** | 956 | **81.9%** | 74.6% |

**Calibrated with training votes.** We fit `p = sigmoid(0.45·logit(p_judge) +0.01)` to the human vote shares of the 349 training pairs, then applied it unchanged to the test task. Test log-loss goes from 0.610 to **0.563**, and agreement with the majority stays at 82.2%. The confidence now reads like a vote share:

| Judge confidence (calibrated) | Pairs | Share of humans who agreed with the judge |
|---|---:|---:|
| 50–60% | 119 | 50.3% |
| 60–70% | 162 | 63.3% |
| 70–80% | 264 | 81.9% |
| 80–100% | 62 | 86.9% |

**A trained judge on the 49k training votes.** Frozen V-JEPA 2 (ViT-L) features for each clip (16 frames), and a linear Bradley-Terry scorer trained on the vote shares of all 4,887 training pairs. Settings were chosen by leave-one-task-out CV on the 7 training tasks (best: `mean` features, C=0.001, LOTO accuracy 75.5%); the test task was scored once.

| Judge | All 607 test pairs | Without X-VLA | Pairs where the policy prior is wrong | Cost / speed per pair |
|---|---:|---:|---:|---|
| Policy identity only (no video) | 65.6% | 46.0% | 0% | – |
| V-JEPA 2 + linear head (trained) | **81.3%** | 73.9% | 68.3% | free, ~1 s on a laptop |
| Muse Spark 1.3 (zero-shot) | **82.6%** | 77.8% | 73.9% | ~$0.07, ~40 s |
| Single human vs. other 9 | 77.4% | – | – | – |

The *policy identity* row predicts every pair from each policy's training-task Elo without watching anything. Both judges beat it by a wide margin, and stay well above chance on the 199 pairs where the usually-weaker policy actually won. So they are judging the run, not recognising the robot.

Muse stays ahead overall. On the 7 training tasks, V-JEPA averages 75.5% (each task scored by a model trained on the other six), against 80.9% for Muse on its sample, so the small gap on the test task partly reflects that `cup_stacking` suits V-JEPA. The trained judge is the cheap first-pass filter, and Muse is the accurate judge that explains itself.
