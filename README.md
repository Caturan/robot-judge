# Robot Judge

**An AI judge that watches two real robot runs and tells you which one did the task better, and why.** It is built on Meta's Muse models and checked against 6,070 human votes on a robot task it never saw.

Checking whether a robot policy improved still means people watching its runs. That is slow, expensive and doesn't scale. Robot Judge turns each run into a storyboard, asks a Muse model to compare the two runs, and returns a verdict, a confidence and a one-sentence reason. We measure it the way you would measure a new human annotator: how often it agrees with the crowd.

## Results

Test set: the held-out `cup_stacking` task from [robot-policy-human-preferences-55k](https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k). It has 607 pairs of real bimanual robot runs, 5 VLA policies, and exactly **10 human votes per pair**. No prompt or design decision was tuned on this task.

<!-- RESULTS -->
_Final numbers are filled in by `src/evaluate.py` once the full run completes._
<!-- /RESULTS -->

**How to read this.** *Agreement with the majority* is the share of non-tied pairs where the judge picks the side most of the 10 humans picked. The **human ceiling** applies the same test to each annotator: how often does one person agree with the majority of the other 9? A judge above that line agrees with the crowd more reliably than a typical single human rater does.

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
scripts/download.sh                               # needs HF access to the gated dataset
.venv/bin/python scripts/extract_clips.py         # parquet -> data/clips/*.mp4
.venv/bin/python src/evaluate.py --model muse-spark-1.3 --workers 12
.venv/bin/python src/evaluate.py --model muse-wonder-1.0 --limit 150 --max-tokens 16000
.venv/bin/python app.py                           # demo UI at http://127.0.0.1:7860
```

| File | What it does |
|---|---|
| [`src/judge.py`](src/judge.py) | Storyboard rendering, the prompt, both-orders judging, resumable parallel runner |
| [`src/metrics.py`](src/metrics.py) | Human ceiling, agreement / log-loss / correlation, Bradley-Terry Elo |
| [`src/evaluate.py`](src/evaluate.py) | Runs a model on a split and writes `results/*_report.json` |
| [`app.py`](app.py) | Gradio demo: pick a pair, see the verdict and reason, then reveal the human vote |
| [`results/`](results/) | Raw verdicts (JSONL) and reports for every run |

## Next steps

- **A trained judge.** Encode clips with Meta's V-JEPA 2 video encoder and train a Bradley-Terry head on the 48,870 training votes from the other 7 tasks. The Muse judge would stay as the explainer.
- **Use the full video.** Add left and right wrist cameras and pass more frames for long tasks.
- **Calibration.** Fit the confidence to the human vote share so "80% sure" means 8 out of 10 people.
- **Closed loop.** Use the judge as a cheap reward signal to rank policy checkpoints during training.

## Credits

Dataset: [Datapoint AI](https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k) (CC-BY-4.0). Robot clips are from [Ai2](https://huggingface.co/allenai) and [Jiafei1224](https://huggingface.co/Jiafei1224) LeRobot evaluation recordings (Apache-2.0). Models: Meta Muse via the Meta Model API.
