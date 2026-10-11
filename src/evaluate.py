"""
Хронологічне розбиття, бейзлайни та метрики для задачі A.

Розбиття (T = кінець сітки, однаковий для всіх h):
  train : [REFORM_END, T−56д)   val : [T−56д, T−28д)   test : [T−28д, T)
Purge: рядок потрапляє в сплит, лише якщо його вікно цілі (t, t+h] не виходить
за межу сплиту — інакше мітка train «бачить» години val/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

H = pd.Timedelta(hours=1)
EPS = 1e-4


# ---------------------------------------------------------------- split
def chrono_split(df: pd.DataFrame, h: int, test_days: int = 28, val_days: int = 28):
    T = df["hour_utc"].max() + h * H                     # останній origin = T − h
    test_start = T - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    t, t_end = df["hour_utc"], df["hour_utc"] + h * H     # ціль займає рядки [t, t+h)
    masks = {
        "train": t_end <= val_start,
        "val": (t >= val_start) & (t_end <= test_start),
        "test": t >= test_start,
    }
    bounds = {"val_start": val_start, "test_start": test_start, "end": T}
    return masks, bounds


# ---------------------------------------------------------------- baselines
def markov_fit(train: pd.DataFrame, m: float = 5.0) -> pd.DataFrame:
    """
    P(target | oblast, hour_t, active_now), оцінене лише на train.
    m-оцінка зі зсувом до P(target | oblast, active_now): клітинки з малим n не дають 0/1.
    """
    parent = train.groupby(["oblast", "active_now"], observed=True)["target"].mean().rename("p_parent")
    cell = train.groupby(["oblast", "hour_t", "active_now"], observed=True)["target"].agg(["sum", "count"])
    cell = cell.join(parent, on=["oblast", "active_now"])
    cell["p"] = (cell["sum"] + m * cell["p_parent"]) / (cell["count"] + m)
    return cell[["p"]]


def markov_predict(table: pd.DataFrame, df: pd.DataFrame, fallback: float) -> np.ndarray:
    p = df.join(table, on=["oblast", "hour_t", "active_now"])["p"]
    return p.fillna(fallback).to_numpy()


# ---------------------------------------------------------------- metrics
def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1])}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, рівні за ймовірністю біни."""
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / len(y)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    d = pd.DataFrame({"bin": idx, "y": y, "p": p})
    return d.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).reset_index()


SLICES = {
    "all": lambda d: np.ones(len(d), dtype=bool),
    "onset": lambda d: (d["active_now"] == 0).to_numpy(),   # тривоги в момент t немає
}


def evaluate(df: pd.DataFrame, preds: dict[str, np.ndarray],
             ref_clim: str = "clim", ref_markov: str = "markov28") -> pd.DataFrame:
    """
    Brier / log-loss / BSS / ECE у зрізах all та onset.
    bss_markov рахується відносно онлайн-Markov (markov28) — найсильнішого бейзлайну.
    """
    rows = []
    y_all = df["target"].to_numpy()
    for sl, fn in SLICES.items():
        m = fn(df)
        y = y_all[m]
        ref = {k: _metrics(y, preds[k][m])["brier"] for k in (ref_clim, ref_markov)}
        for name, p_all in preds.items():
            p = p_all[m]
            r = _metrics(y, p)
            rows.append({
                "slice": sl, "model": name, **r,
                "bss_clim": 1 - r["brier"] / ref[ref_clim],
                "bss_markov": 1 - r["brier"] / ref[ref_markov],
                "ece": ece(y, p), "base_rate": y.mean(), "n": int(m.sum()),
            })
    return pd.DataFrame(rows)


def block_bootstrap_diff(df: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray,
                         mask: np.ndarray | None = None, n: int = 2000, seed: int = 0) -> tuple:
    """
    ΔBrier = Brier(A) − Brier(B) з 95% CI; ресемплінг цілих днів (UTC) зберігає
    автокореляцію всередині доби. Від'ємне Δ = A краща.
    """
    mask = np.ones(len(df), dtype=bool) if mask is None else mask
    y = df["target"].to_numpy()[mask]
    d = (np.clip(p_a[mask], EPS, 1 - EPS) - y) ** 2 - (np.clip(p_b[mask], EPS, 1 - EPS) - y) ** 2
    day = df["hour_utc"].dt.floor("D").to_numpy()[mask]
    agg = pd.DataFrame({"day": day, "d": d}).groupby("day")["d"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agg), size=(n, len(agg)))
    boots = agg[idx, 0].sum(1) / agg[idx, 1].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
