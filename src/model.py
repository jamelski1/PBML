"""PowerballTransformer — a small Transformer encoder over recent draw history.

EDUCATIONAL PROJECT. Powerball draws are independent random events; no model
can genuinely predict them. This architecture exists to teach the end-to-end
ML workflow (sequence encoding, training on GPU, evaluation, deployment).

Input encoding
--------------
Each historical draw is a 95-dim multi-hot vector:
  - dims 0..68  : the five white balls (1-69), one-hot x5 summed
  - dims 69..94 : the Powerball (1-26)
A training example is a window of the last `window` draws: (window, 95).

Output
------
  - white_logits: (batch, 69) — scores for each white ball number
  - pb_logits:    (batch, 26) — scores for the Powerball
"""

import json

import torch
import torch.nn as nn

N_WHITE = 69   # white balls are drawn from 1..69  (rules since Oct 2015)
N_PB = 26      # the Powerball is drawn from 1..26
DRAW_DIM = N_WHITE + N_PB  # 95


class PowerballTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
                 window=32, dropout=0.1):
        super().__init__()
        self.config = dict(d_model=d_model, nhead=nhead, num_layers=num_layers,
                           dim_feedforward=dim_feedforward, window=window,
                           dropout=dropout)
        self.window = window

        # Project each 95-dim multi-hot draw vector into the model dimension.
        self.input_proj = nn.Linear(DRAW_DIM, d_model)

        # Learned positional embeddings — one per slot in the history window.
        self.pos_emb = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Two prediction heads: 69 white-ball logits and 26 Powerball logits.
        self.head_white = nn.Linear(d_model, N_WHITE)
        self.head_pb = nn.Linear(d_model, N_PB)

    def forward(self, x):
        # x: (batch, window, 95) float multi-hot
        h = self.input_proj(x) + self.pos_emb[:, : x.size(1)]
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))  # mean-pool over the history window
        return self.head_white(h), self.head_pb(h)

    # -- persistence helpers (used by the notebook and the Gradio Space) -----

    def save(self, dir_path):
        import os
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.state_dict(), f"{dir_path}/pytorch_model.bin")
        with open(f"{dir_path}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, dir_path, map_location="cpu"):
        with open(f"{dir_path}/config.json") as f:
            config = json.load(f)
        model = cls(**config)
        state = torch.load(f"{dir_path}/pytorch_model.bin",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


def encode_draw(whites, pb):
    """Encode one draw (list of five 1-based white balls, 1-based powerball)
    into a 95-dim multi-hot float tensor."""
    v = torch.zeros(DRAW_DIM)
    for w in whites:
        v[w - 1] = 1.0
    v[N_WHITE + pb - 1] = 1.0
    return v


@torch.no_grad()
def sample_play(model, history, temperature=1.0, generator=None):
    """Sample a playable ticket (5 white balls + Powerball) from the model.

    history: (window, 95) tensor of the most recent draws.
    Returns (sorted list of 5 white ints, powerball int).
    """
    white_logits, pb_logits = model(history.unsqueeze(0))
    white_p = torch.softmax(white_logits[0] / temperature, dim=0)
    pb_p = torch.softmax(pb_logits[0] / temperature, dim=0)
    # Sample 5 distinct white balls, weighted by the model's probabilities.
    whites = torch.multinomial(white_p, 5, replacement=False, generator=generator)
    pb = torch.multinomial(pb_p, 1, generator=generator)
    return sorted((whites + 1).tolist()), int(pb.item()) + 1
