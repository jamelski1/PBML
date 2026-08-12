# pbml — Powerball ML 🎱

An **educational, end-to-end machine-learning project** built on historical Powerball
data: real data ingestion, exploratory analysis, statistical testing, Transformer
training on a Colab GPU, honest backtesting, and publishing to the Hugging Face Hub.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jamelski1/pbml/blob/claude/powerball-prediction-ml-fxhi8i/notebooks/powerball_ml.ipynb)

## ⚠️ The honest part

Powerball draws are **independent random events** — no model can genuinely predict
them, and this project proves it to itself: a chi-square test shows the historical
draws are statistically uniform, and a walk-forward backtest shows the trained
Transformer matches numbers at exactly the rate of random guessing (~0.36 white
balls per draw, ~3.8% Powerball hits). Producing and trusting that negative result
is the core lesson. **Never pay anyone for lottery predictions.**

## What's here

| Path | What it is |
|---|---|
| [`notebooks/powerball_ml.ipynb`](notebooks/powerball_ml.ipynb) | The main event — full pipeline, runs top-to-bottom in Google Colab |
| [`src/model.py`](src/model.py) | `PowerballTransformer` — the model class, shared by notebook and app |
| [`app/app.py`](app/app.py) | Gradio number-generator app, ready for Hugging Face Spaces |
| [`requirements.txt`](requirements.txt) | Dependencies for local runs / the Space |

## Quick start

1. Click the **Open in Colab** badge above
2. `Runtime → Change runtime type → T4 GPU` (free) — or A100/L4 with Colab Pro
3. `Runtime → Run all`

Data comes from [NY State Open Data](https://data.ny.gov/Government-Finance/Lottery-Powerball-Winning-Numbers-Beginning-2010/d6yy-54nr)
(every Powerball draw, no API key). The notebook filters to the current-rules era
(Oct 2015+: white balls 1–69, Powerball 1–26) since earlier eras used different
number ranges.

## The pipeline

1. **Load & clean** — download, parse, era-filter, sanity-check
2. **EDA** — frequencies, hot/cold numbers, gap analysis, pair co-occurrence
3. **Chi-square test** — are the "patterns" real? (No.)
4. **Dataset** — 95-dim multi-hot draw encoding, sliding windows, chronological split
5. **Model** — small Transformer encoder (~426K params), dual heads (white + PB)
6. **Training** — GPU with mixed precision, train/val curves
7. **Backtest** — walk-forward vs. frequency-picker and random baselines
8. **Generate** — sample playable tickets from the model
9. **Publish** — push weights + honest model card to the Hugging Face Hub

## Roadmap

- [x] Colab training notebook
- [ ] Publish trained model to the HF Hub (flip `PUSH_TO_HUB` in the notebook)
- [ ] Deploy the Gradio app to HF Spaces (`app/app.py` is ready)

## License

MIT — educational use encouraged.
