"""pbml experiment harness — a rig for systematically hunting for signal in
Powerball history, with a permutation control as the built-in referee.

The scientific contract
-----------------------
Every experiment is run TWICE:
  1. on the real data (chronological order preserved)
  2. on a "permuted" copy where the past->future pairing is destroyed but each
     number's marginal frequency is untouched

If a model beats random on BOTH, it's memorizing noise or leaking -> reject.
If it beats random on the REAL data but NOT the permuted data, that gap is a
candidate signal. We quantify the gap and its significance so "did it work?"
becomes a measurement, not a vibe.

Nothing in here can make the lottery predictable. What it CAN do is tell us,
honestly and reproducibly, whether any given approach found conditional
structure — and make a real signal impossible to miss if one ever appears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_WHITE, N_PB = 69, 26
RULES_CHANGE = "2015-10-07"


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_draws(url_or_path="https://data.ny.gov/api/views/d6yy-54nr/rows.csv"
                           "?accessType=DOWNLOAD", era_filter=True):
    """Return a clean DataFrame with columns w1..w5 (sorted), pb, and Draw Date."""
    raw = pd.read_csv(url_or_path)
    df = raw.copy()
    df["Draw Date"] = pd.to_datetime(df["Draw Date"])
    nums = df["Winning Numbers"].str.split(expand=True).astype(int)
    df[["w1", "w2", "w3", "w4", "w5"]] = np.sort(nums.iloc[:, :5].values, axis=1)
    df["pb"] = nums.iloc[:, 5]
    if era_filter:
        df = df[df["Draw Date"] >= RULES_CHANGE]
    df = df.sort_values("Draw Date").reset_index(drop=True)
    return df


WHITE_COLS = ["w1", "w2", "w3", "w4", "w5"]


# --------------------------------------------------------------------------- #
#  Targets — what we ask the model to predict about draw t given draws < t
# --------------------------------------------------------------------------- #
def make_target(df, kind, number=None):
    """Return (y, n_classes, task) for a given target framing.

    task is 'multilabel' (BCE over N_WHITE) or 'multiclass' (CE over n_classes).
    """
    whites = df[WHITE_COLS].values
    if kind == "white_multihot":                       # the honest hard target
        y = np.zeros((len(df), N_WHITE), dtype=np.float32)
        for i, row in enumerate(whites):
            y[i, row - 1] = 1.0
        return y, N_WHITE, "multilabel"
    if kind == "pb":                                   # powerball 1..26
        return df["pb"].values - 1, N_PB, "multiclass"
    if kind == "sum_bucket":                           # bell-curved marginal!
        s = whites.sum(axis=1)                         # 15..345, ~normal
        edges = np.linspace(s.min(), s.max() + 1, 11)  # 10 buckets
        return np.digitize(s, edges) - 1, 10, "multiclass"
    if kind == "odd_count":                            # 0..5 odds among 5 balls
        return (whites % 2 == 1).sum(axis=1), 6, "multiclass"
    if kind == "high_count":                           # count of balls > 34
        return (whites > 34).sum(axis=1), 6, "multiclass"
    if kind == "contains":                             # does draw t contain N?
        assert number is not None
        return (whites == number).any(axis=1).astype(np.int64), 2, "multiclass"
    raise ValueError(f"unknown target kind: {kind}")


# --------------------------------------------------------------------------- #
#  Features — what the model sees about the past
# --------------------------------------------------------------------------- #
def make_features(df, window, extras=()):
    """Return X of shape (n_examples, window, feat_dim) plus aligned index.

    Base features per past draw: 95-dim multi-hot (69 white + 26 pb).
    Optional extras broadcast across the window as extra channels:
      'dow'   day-of-week / 6         (calendar structure, if any)
      'month' month / 11
      'gap'   draws since each ball last seen (69-dim) -- recency features
    """
    n = len(df)
    base = np.zeros((n, N_WHITE + N_PB), dtype=np.float32)
    for i, row in enumerate(df[WHITE_COLS + ["pb"]].values):
        base[i, row[:5] - 1] = 1.0
        base[i, N_WHITE + row[5] - 1] = 1.0

    channels = [base]
    if "gap" in extras:
        gap = np.zeros((n, N_WHITE), dtype=np.float32)
        last = {k: -1 for k in range(N_WHITE)}
        for i, row in enumerate(df[WHITE_COLS].values):
            for b in range(N_WHITE):
                gap[i, b] = (i - last[b]) if last[b] >= 0 else i
            for w in row:
                last[w - 1] = i
        channels.append(gap / 50.0)
    if "dow" in extras:
        dow = (df["Draw Date"].dt.dayofweek.values / 6.0).astype(np.float32)
        channels.append(dow[:, None])
    if "month" in extras:
        mon = ((df["Draw Date"].dt.month.values - 1) / 11.0).astype(np.float32)
        channels.append(mon[:, None])

    feat = np.concatenate(channels, axis=1)
    X = np.stack([feat[i - window:i] for i in range(window, n)])
    return X, np.arange(window, n)


# --------------------------------------------------------------------------- #
#  Permutation control — the referee
# --------------------------------------------------------------------------- #
def physical_bias_test(df):
    """The correct instrument for the physical-bias hypothesis: a direct
    frequency test over ALL draws (no windowing, no split -> maximum power).

    A worn/heavy ball shows up as a number appearing more often than 5/69 (white)
    or 1/26 (Powerball) -- a marginal effect. We run:
      - a chi-square goodness-of-fit over all numbers, and
      - a per-number two-sided binomial test, Bonferroni-corrected for the
        multiple comparisons (testing 69 + 26 numbers inflates false positives).

    Returns a dict with the overall p-values and any numbers whose corrected
    p < 0.05 (i.e. survive the multiple-comparison correction).
    """
    from scipy import stats

    n = len(df)
    out = {"n_draws": n, "findings": []}
    for label, cols, k, hi in [("white", WHITE_COLS, 5, N_WHITE),
                               ("pb", ["pb"], 1, N_PB)]:
        vals = df[cols].values.ravel()
        counts = np.bincount(vals, minlength=hi + 1)[1:]  # counts for 1..hi
        p_each = k / hi                                    # P(number in a draw)
        expected = n * p_each
        chi2, p_chi = stats.chisquare(counts)
        # per-number two-sided binomial, Bonferroni over `hi` tests
        binom_p = np.array([stats.binomtest(int(c), n, p_each).pvalue
                            for c in counts])
        corrected = np.minimum(binom_p * hi, 1.0)
        survivors = [(i + 1, int(counts[i]), round(float(corrected[i]), 4))
                     for i in np.argsort(corrected)[:5] if corrected[i] < 0.05]
        out[label] = {"chi2": round(float(chi2), 2),
                      "chi2_p": round(float(p_chi), 4),
                      "expected_per_number": round(float(expected), 1),
                      "most_extreme": [(int(np.argmax(counts) + 1),
                                        int(counts.max())),
                                       (int(np.argmin(counts) + 1),
                                        int(counts.min()))],
                      "significant_after_correction": survivors}
        out["findings"] += [f"{label}: {n}"] if survivors else []
    out["any_signal"] = bool(out["white"]["significant_after_correction"]
                             or out["pb"]["significant_after_correction"])
    return out


def permute_targets(y, seed):
    """Destroy the feature->target (past->future) relationship while keeping the
    target's marginal distribution identical. This is our negative control."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return y[perm]


# --------------------------------------------------------------------------- #
#  Chronological split
# --------------------------------------------------------------------------- #
def chrono_split(n, train=0.8, val=0.1):
    n_tr, n_va = int(n * train), int(n * val)
    return (slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n))
