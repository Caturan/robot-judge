"""Robot Judge demo: two robot runs in, verdict + explanation out.

  .venv/bin/python app.py
"""
import base64
import os
import sys

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import judge  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
pairs = pd.read_parquet(f"{RAW}/pairs/test.parquet")
TASK = pd.read_parquet(f"{RAW}/tasks/test.parquet").instruction_shown.iloc[0]
NAMES = dict(pd.read_parquet(f"{RAW}/models/models.parquet")[["model_id", "display_name"]].values)
MODEL = "muse-spark-1.3"


def clip_path(key):
    return os.path.join(judge.CLIPS, key.replace("/", "__") + ".mp4")


def sb_image(key):
    path = os.path.join("/tmp", key.replace("/", "__") + "_sb.jpg")
    with open(path, "wb") as f:
        f.write(base64.b64decode(judge.storyboard(key)))
    return path


def pick(pair_key):
    r = pairs.set_index("pair_key").loc[pair_key]
    return clip_path(r.video_a_key), clip_path(r.video_b_key), TASK, ""


def run_judge(pair_key, task):
    r = pairs.set_index("pair_key").loc[pair_key]
    out = judge.judge_pair(MODEL, task, r.video_a_key, r.video_b_key)
    if out["p_a"] is None:
        return "Judge failed, try again.", None, None
    p = out["p_a"]
    side = "A" if p > 0.5 else "B" if p < 0.5 else "neither (tie)"
    v = out["fwd"] if "winner" in out["fwd"] else out["rev"]
    md = (
        f"## Judge: Run {side} did better ({max(p, 1 - p):.0%} confident)\n\n"
        f"**Why:** {v['reason']}\n\n"
        f"- Run A: {v['run_1_summary'] if v is out['fwd'] else v['run_2_summary']}\n"
        f"- Run B: {v['run_2_summary'] if v is out['fwd'] else v['run_1_summary']}\n\n"
        f"Order check: A-first says **{out['fwd'].get('winner', '?')}**, "
        f"B-first says **{out['rev'].get('winner', '?')}**\n\n"
        f"---\n**10 humans voted:** A {r.votes_a} – {r.votes_b} B  \n"
        f"_Revealed: A = {NAMES[r.model_a]}, B = {NAMES[r.model_b]}_"
    )
    return md, sb_image(r.video_a_key), sb_image(r.video_b_key)


examples = pairs[(pairs.votes_a >= 8) | (pairs.votes_a <= 2)].sample(40, random_state=1).pair_key.tolist()

with gr.Blocks(title="Robot Judge") as demo:
    gr.Markdown(
        "# Robot Judge\nWhich robot did the task better, and why? "
        f"Zero-shot judge on Meta `{MODEL}`, checked against 10 human votes per pair. "
        "Pairs come from the held-out `cup_stacking` task."
    )
    with gr.Row():
        pk = gr.Dropdown(examples, value=examples[0], label="Pair")
        task = gr.Textbox(TASK, label="Task")
    with gr.Row():
        va, vb = gr.Video(label="Run A"), gr.Video(label="Run B")
    btn = gr.Button("Judge", variant="primary")
    verdict = gr.Markdown()
    with gr.Row():
        sa, sb = gr.Image(label="What the judge saw: A"), gr.Image(label="What the judge saw: B")
    pk.change(pick, pk, [va, vb, task, verdict])
    demo.load(pick, pk, [va, vb, task, verdict])
    btn.click(run_judge, [pk, task], [verdict, sa, sb])

if __name__ == "__main__":
    demo.launch()
