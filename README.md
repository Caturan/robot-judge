# Robot Judge

**An AI judge that watches two real robot runs and tells you which one did the task better, and why.** It is built on Meta's Muse models and checked against 6,070 human votes on a robot task it never saw.

**[Watch the 3-minute demo video](https://www.loom.com/share/cd6e97a0aca344d0afbce0f3d25c90f4)**

Checking whether a robot policy improved still means people watching its runs. That is slow, expensive and doesn't scale. Robot Judge turns each run into a storyboard, asks a Muse model to compare the two runs, and returns a verdict, a confidence and a one-sentence reason. We measure it the way you would measure a new human annotator: how often it agrees with the crowd.

## Results

Test set: the held-out `cup_stacking` task from [robot-policy-human-preferences-55k](https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k). It has 607 pairs of real bimanual robot runs, 5 VLA policies, and exactly **10 human votes per pair**. No prompt or design decision was tuned on this task.

<!-- RESULTS -->
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
<!-- /RESULTS -->

**How to read this.** *Agreement with the majority* is the share of non-tied pairs where the judge picks the side most of the 10 humans picked. The **human ceiling** applies the same test to each annotator: how often does one person agree with the majority of the other 9? A judge above that line agrees with the crowd more reliably than a typical single human rater does.

Two caveats. First, the judge is scored against the majority of all 10 votes, while each human is scored against the other 9, so the reference for the judge is slightly less noisy. Second, the top four policies on this task are within about 20 Elo of each other on human votes alone, so swaps among them are within noise. The clear gap is between X-VLA and everyone else.

## How it works

```
run A.mp4 ─┐                                 ┌─ verdict: A or B
           ├─ storyboard: 8 frames + FINAL ──┼─ confidence 0.5–1.0
run B.mp4 ─┘   (one tiled image per run)     └─ reason + per-run summary
                         │
              Muse (Meta Model API), judged twice: A-first and B-first
                         │
              p(A better) = mean of both orders  → cancels position bias
```

- **Storyboards, not raw video.** Each clip becomes a 3×3 grid: 8 evenly spaced frames from the run plus the final frame, each with a timestamp. One image per run keeps cost at roughly 1.5k input tokens per comparison. The final frame shows the end state, which is what humans weigh most.
- **The prompt follows how humans judge.** It credits completion first, then partial progress, then fewer mistakes (drops, knocking things over, stalling). The dataset card shows annotators credit partial progress even when every run fails.
- **Both orders.** Every pair is judged with A shown first and with B shown first. The two answers are averaged into a probability, and we report how often the two orders agree (*order consistency*).
- **Honest evaluation.** The test split is a whole unseen task. Failed API calls are retried and counted, never dropped. In a pilot, dropping the failed pairs inflated `muse-wonder-1.0` from 89% to 96%, because the pairs that failed were the hard ones.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "META_API_KEY=..." > .env                    # Meta Model API key
scripts/download.sh --train-videos                # needs HF access to the gated dataset
.venv/bin/python scripts/extract_clips.py         # parquet -> data/clips/*.mp4
.venv/bin/python src/evaluate.py --model muse-spark-1.3 --workers 12
.venv/bin/python src/evaluate.py --model muse-wonder-1.0 --limit 150 --max-tokens 16000
.venv/bin/python src/evaluate.py --split train --model muse-spark-1.3 --limit 350   # per-task + calibration

# trained judge (Python 3.10+; torch + transformers)
python3.11 -m venv .venv-vjepa && .venv-vjepa/bin/pip install -r requirements-vjepa.txt
.venv-vjepa/bin/python src/vjepa_features.py      # ~20 min on an M-series Mac
.venv-vjepa/bin/python src/train_vjepa.py

.venv/bin/python src/compare.py                   # rebuilds the results tables in this README
.venv/bin/python app.py                           # demo UI at http://127.0.0.1:7860
```

| File | What it does |
|---|---|
| [`src/judge.py`](src/judge.py) | Storyboard rendering, the prompt, both-orders judging, resumable parallel runner |
| [`src/metrics.py`](src/metrics.py) | Human ceiling, agreement / log-loss / correlation, Bradley-Terry Elo |
| [`src/evaluate.py`](src/evaluate.py) | Runs a model on a split and writes `results/*_report.json` |
| [`src/compare.py`](src/compare.py) | Builds every results table in this README from the raw verdicts |
| [`src/vjepa_features.py`](src/vjepa_features.py) | Encodes every clip with frozen V-JEPA 2 (runs on Apple-silicon GPU) |
| [`src/train_vjepa.py`](src/train_vjepa.py) | Trains the Bradley-Terry head on the training votes, with leave-one-task-out model selection |
| [`app.py`](app.py) | Gradio demo: pick a pair, see the verdict and reason, then reveal the human vote |
| [`results/`](results/) | Raw verdicts (JSONL) and reports for every run |

## Next steps

- **A stronger trained judge.** Fine-tune V-JEPA 2 end to end, or use an attentive probe over all patch tokens instead of a linear head on pooled features.
- **Cascade.** Let V-JEPA settle the clear-cut pairs for free and send only the uncertain ones to Muse.
- **Use the full video.** Add left and right wrist cameras and pass more frames for long tasks.
- **Closed loop.** Use the judge as a cheap reward signal to rank policy checkpoints during training.

## Credits

Dataset: [Datapoint AI](https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k) (CC-BY-4.0). Robot clips are from [Ai2](https://huggingface.co/allenai) and [Jiafei1224](https://huggingface.co/Jiafei1224) LeRobot evaluation recordings (Apache-2.0). Models: Meta Muse via the Meta Model API.
