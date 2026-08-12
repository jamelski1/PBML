"""Model zoo + unified runner for the pbml experiment harness.

run_experiment(cfg) trains one model on the real data AND on a permutation
control, then reports the gap between them. A positive, significant gap on the
held-out test set is our definition of "found something."

Model backends: 'freq', 'markov', 'mlp', 'lstm', 'gru', 'tcn', 'transformer'.
(XGBoost/sklearn can be slotted in the same way — see 'gbdt'.)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import (chrono_split, make_features, make_target,
                         permute_targets, N_WHITE)


# --------------------------------------------------------------------------- #
#  Deep model backends
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, feat_dim, window, n_out, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim * window, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class _RNN(nn.Module):
    def __init__(self, feat_dim, n_out, hidden=128, layers=2, kind="lstm",
                 dropout=0.1):
        super().__init__()
        rnn = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn(feat_dim, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _TCN(nn.Module):
    """Dilated temporal conv net — the strong sequence baseline."""
    def __init__(self, feat_dim, n_out, ch=64, levels=4, k=3, dropout=0.1):
        super().__init__()
        layers, c_in = [], feat_dim
        for i in range(levels):
            d = 2 ** i
            layers += [nn.Conv1d(c_in, ch, k, padding=(k - 1) * d, dilation=d),
                       nn.ReLU(), nn.Dropout(dropout)]
            c_in = ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(ch, n_out)

    def forward(self, x):                 # x: (B, W, F)
        h = self.net(x.transpose(1, 2))   # (B, C, W')
        return self.head(h.mean(dim=2))


class _Transformer(nn.Module):
    def __init__(self, feat_dim, window, n_out, d_model=128, nhead=4,
                 layers=3, ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers,
                                         enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.size(1)]
        h = self.norm(self.enc(h).mean(dim=1))
        return self.head(h)


def _build(cfg, feat_dim, window, n_out):
    m = cfg["model"]
    if m == "mlp":
        return _MLP(feat_dim, window, n_out)
    if m in ("lstm", "gru"):
        return _RNN(feat_dim, n_out, kind=m)
    if m == "tcn":
        return _TCN(feat_dim, n_out)
    if m == "transformer":
        return _Transformer(feat_dim, window, n_out)
    raise ValueError(f"unknown model {m}")


# --------------------------------------------------------------------------- #
#  Non-deep baselines (no training loop)
# --------------------------------------------------------------------------- #
def _baseline_metrics(cfg, df, y, n_classes, task, splits):
    """freq: always predict the training-set marginal mode.
    markov: predict class most likely to follow the previous draw's class."""
    tr, va, te = splits
    if cfg["model"] == "freq":
        if task == "multilabel":
            counts = y[tr].sum(0)
            top5 = set(np.argsort(counts)[-5:])
            hits = [len(top5 & set(np.nonzero(r)[0])) for r in y[te]]
            return {"test_metric": float(np.mean(hits)), "metric": "white_hits"}
        mode = np.bincount(y[tr], minlength=n_classes).argmax()
        acc = float((y[te] == mode).mean())
        return {"test_metric": acc, "metric": "accuracy"}
    raise ValueError("markov handled inline")


# --------------------------------------------------------------------------- #
#  The runner
# --------------------------------------------------------------------------- #
def _train_eval(model, Xtr, ytr, Xte, yte, task, n_classes, device,
                epochs, lr):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = (nn.BCEWithLogitsLoss() if task == "multilabel"
               else nn.CrossEntropyLoss())
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device)).cpu()
    if task == "multilabel":
        top5 = logits.topk(5, dim=1).indices
        hits = [len(set(p.tolist()) & set(torch.nonzero(a).squeeze(1).tolist()))
                for p, a in zip(top5, yte)]
        return float(np.mean(hits)), "white_hits"
    acc = float((logits.argmax(1) == yte).float().mean())
    return acc, "accuracy"


def run_experiment(df, cfg, device=None, epochs=40, lr=3e-4, seed=0,
                   verbose=True):
    """Run one config on real data and on the permutation control.

    cfg keys: model, target, window, extras (tuple), number (for 'contains').
    Returns a dict with real vs. permuted test metric and the gap.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    window = cfg.get("window", 32)
    extras = cfg.get("extras", ())
    y_all, n_classes, task = make_target(df, cfg["target"],
                                         cfg.get("number"))

    def evaluate(y_full):
        X, idx = make_features(df, window, extras)
        y = y_full[idx]
        splits = chrono_split(len(X))
        if cfg["model"] == "freq":
            return _baseline_metrics(cfg, df, y, n_classes, task, splits)
        tr, _, te = splits
        Xt = torch.tensor(X)
        yt = (torch.tensor(y) if task == "multiclass"
              else torch.tensor(y, dtype=torch.float32))
        model = _build(cfg, X.shape[2], window, n_classes)
        metric, name = _train_eval(model, Xt[tr], yt[tr], Xt[te], yt[te],
                                   task, n_classes, device, epochs, lr)
        return {"test_metric": metric, "metric": name}

    real = evaluate(y_all)
    perm = evaluate(permute_targets(y_all, seed))
    gap = real["test_metric"] - perm["test_metric"]
    result = {**cfg, "metric": real["metric"],
              "real": round(real["test_metric"], 4),
              "permuted": round(perm["test_metric"], 4),
              "gap": round(gap, 4)}
    if verbose:
        print(f"{cfg['model']:>11} / {cfg['target']:<13} "
              f"real={result['real']:.4f}  perm={result['permuted']:.4f}  "
              f"gap={result['gap']:+.4f}  [{real['metric']}]")
    return result


def permutation_test(df, cfg, n_perms=30, device=None, epochs=40, lr=3e-4,
                     verbose=True):
    """The referee, done rigorously. Trains once on the real data, then n_perms
    times on independently shuffled copies, building the null distribution of
    the metric. Returns real metric, the null mean/std, a z-score, and a
    one-sided p-value = P(null >= real).

    p < 0.01 on a held-out target we didn't tune against = our pre-registered
    definition of a genuine signal.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    window = cfg.get("window", 32)
    extras = cfg.get("extras", ())
    y_all, n_classes, task = make_target(df, cfg["target"], cfg.get("number"))

    def eval_one(y_full):
        X, idx = make_features(df, window, extras)
        y = y_full[idx]
        splits = chrono_split(len(X))
        if cfg["model"] == "freq":
            return _baseline_metrics(cfg, df, y, n_classes, task,
                                     splits)["test_metric"]
        tr, _, te = splits
        Xt = torch.tensor(X)
        yt = (torch.tensor(y) if task == "multiclass"
              else torch.tensor(y, dtype=torch.float32))
        model = _build(cfg, X.shape[2], window, n_classes)
        return _train_eval(model, Xt[tr], yt[tr], Xt[te], yt[te], task,
                           n_classes, device, epochs, lr)[0]

    real = eval_one(y_all)
    null = np.array([eval_one(permute_targets(y_all, s))
                     for s in range(1, n_perms + 1)])
    mu, sd = null.mean(), null.std() + 1e-9
    z = (real - mu) / sd
    p = float((null >= real).mean())        # one-sided permutation p-value
    out = {**cfg, "real": round(real, 4), "null_mean": round(float(mu), 4),
           "null_std": round(float(sd), 4), "z": round(float(z), 2),
           "p_value": p, "signal": p < 0.01}
    if verbose:
        flag = "  <-- SIGNAL!" if out["signal"] else ""
        print(f"{cfg['model']:>11} / {cfg['target']:<13} real={out['real']:.4f} "
              f"null={mu:.4f}±{sd:.4f}  z={out['z']:+.2f}  p={p:.3f}{flag}")
    return out
