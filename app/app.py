"""Gradio web interface for pbml — deployable to Hugging Face Spaces.

Loads the trained PowerballTransformer from the HF Hub and generates tickets.
If the model repo isn't published yet (or fails to load), it falls back to
pure uniform random picks — which, per the project's own backtest, performs
identically anyway. That's the joke, and also the truth.

Deploy: create a Gradio Space, add this file + requirements.txt + src/model.py,
and set HF_REPO_ID below (or the MODEL_REPO env var).
"""

import os
import random
import sys

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HF_REPO_ID = os.environ.get("MODEL_REPO", "YOUR_HF_USERNAME/powerball-transformer")

N_WHITE, N_PB = 69, 26

_model = None
_history = None


def _try_load_model():
    """Best-effort: pull weights + recent draw history; None on any failure."""
    global _model, _history
    try:
        import numpy as np
        import pandas as pd
        import torch
        from huggingface_hub import snapshot_download

        from model import PowerballTransformer, encode_draw

        local = snapshot_download(HF_REPO_ID)
        _model = PowerballTransformer.load(local)

        url = "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD"
        df = pd.read_csv(url)
        df["Draw Date"] = pd.to_datetime(df["Draw Date"])
        df = df[df["Draw Date"] >= "2015-10-07"].sort_values("Draw Date")
        nums = df["Winning Numbers"].str.split(expand=True).astype(int)
        window = _model.window
        recent = nums.tail(window).values
        _history = torch.stack(
            [encode_draw(sorted(r[:5]), r[5]) for r in recent])
        return True
    except Exception as e:  # pragma: no cover - network/hub dependent
        print(f"Model unavailable ({e}); using uniform random fallback.")
        _model = _history = None
        return False


def generate(n_tickets, temperature):
    lines = []
    for i in range(int(n_tickets)):
        if _model is not None and _history is not None:
            from model import sample_play
            whites, pb = sample_play(_model, _history, temperature=temperature)
            src = "🤖 model"
        else:
            whites = sorted(random.sample(range(1, N_WHITE + 1), 5))
            pb = random.randint(1, N_PB)
            src = "🎲 uniform random"
        lines.append(
            f"Ticket {i + 1}:  " + "  ".join(f"{w:2d}" for w in whites)
            + f"   ⚡ PB {pb}   ({src})")
    lines.append("")
    lines.append("Every ticket has identical 1-in-292,201,338 jackpot odds. "
                 "The model cannot beat random — see the project backtest.")
    return "\n".join(lines)


DISCLAIMER = """
# 🎱 Powerball ML — educational number generator

**This model cannot predict the lottery.** Powerball draws are independent random
events; the project's walk-forward backtest shows this Transformer ties with random
guessing, exactly as probability theory requires. It exists to demonstrate an
end-to-end ML workflow: data → EDA → training → honest evaluation → deployment.

[Training notebook & source →](https://github.com/jamelski1/pbml)
"""

with gr.Blocks(title="Powerball ML") as demo:
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        n_tickets = gr.Slider(1, 10, value=3, step=1, label="Tickets")
        temperature = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                label="Sampling temperature")
    btn = gr.Button("Generate my numbers 🎟️", variant="primary")
    out = gr.Textbox(label="Your tickets", lines=8, show_copy_button=True)
    btn.click(generate, inputs=[n_tickets, temperature], outputs=out)

if __name__ == "__main__":
    _try_load_model()
    demo.launch()
