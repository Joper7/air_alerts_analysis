"""
Експеримент після діагностики нового режиму (з 07.09.2026).

Діагноз (reports/backtest_wf_h1.parquet, onset):
  до 07.09: AUC моделі 0.688 vs markov28 0.602;  після: 0.647 vs 0.641 — розрізнення зникло.
  Постів @kpszsu у 09.2026 ~6 300 проти ~2 500/міс раніше → абсолютні лічильники TG
  виходять за діапазон, на якому вчились дерева.

Гіпотеза (зафіксована ДО запуску): відносні ознаки + вага свіжих рядків відновлять розрізнення.
  n  — лічильники TG і стартів тривог діляться на їхнє середнє за попередні 28 днів
       (лише минулі origin → без витоку);
  w  — вага рядка 0.5^(вік / 28 днів) у кожному перенавчанні walk-forward;
  nw — обидва. Кандидат на фінал — nw (обраний до запуску); звітуються всі варіанти.

Запуск:  python -m src.regime_fix 1 3
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import CAL_WINDOWS, N_TREES, PRED_START, REFIT_EVERY, online_platt
from src.evaluate import block_bootstrap_diff
from src.features import FEATURE_SETS, make_features
from src.train import LGB_PARAMS, PROC_DIR, REPORT_DIR, _logit, _sigmoid, load_posts

H = pd.Timedelta(hours=1)
NORM_DAYS = 28
HALF_LIFE_DAYS = 28
REGIME = pd.Timestamp("2026-09-07", tz="UTC")
BT_START = pd.Timestamp("2026-03-18", tz="UTC")

BASE = FEATURE_SETS["L1.5GT"]
COUNT_COLS = [c for c in BASE if c.startswith("tg_") and not c.startswith("tg_log")] + \
             ["starts_24h", "starts_7d", "nb_starts_1h", "nb_starts_3h", "n_alert_lasthour", "n_alert_max_6h"]


def add_relative(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """c_rel = c / (середнє c за попередні 28 днів + 0.1), окремо по області; лише минуле (shift 1)."""
    df = df.sort_values(["oblast", "hour_utc"]).copy()
    new = []
    for c in COUNT_COLS:
        mean_past = df.groupby("oblast", observed=True)[c].transform(
            lambda s: s.shift(1).rolling(24 * NORM_DAYS, min_periods=24).mean())
        df[f"{c}_rel"] = df[c] / (mean_past + 0.1)
        new.append(f"{c}_rel")
    cols = [c for c in BASE if c not in COUNT_COLS] + new
    return df.sort_index(), cols


def walk_forward(df: pd.DataFrame, h: int, cols: list[str], weighted: bool) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    refit, end = PRED_START, df["hour_utc"].max() + H
    while refit < end:
        tr = df[df["hour_utc"] + h * H <= refit]
        te_mask = (df["hour_utc"] >= refit) & (df["hour_utc"] < refit + REFIT_EVERY)
        if te_mask.any():
            w = None
            if weighted:
                age = (refit - tr["hour_utc"]).dt.total_seconds() / 86400
                w = 0.5 ** (age / HALF_LIFE_DAYS)
            ds = lgb.Dataset(tr[cols], tr["target"], weight=w,
                             init_score=_logit(tr["markov28"].to_numpy()), categorical_feature=["oblast"])
            booster = lgb.train(LGB_PARAMS, ds, num_boost_round=N_TREES)
            te = df[te_mask]
            out[te_mask] = _sigmoid(booster.predict(te[cols], raw_score=True) + _logit(te["markov28"].to_numpy()))
        refit += REFIT_EVERY
    return out


def run(h: int, hourly, intervals, posts) -> list[dict]:
    kind = "window" if h <= 3 else "point"
    df = make_features(hourly, intervals, h, kind, posts=posts)
    df, rel_cols = add_relative(df)
    variants = {"GT": (BASE, False), "GT_w": (BASE, True), "GT_n": (rel_cols, False), "GT_nw": (rel_cols, True)}
    preds = {"markov28": df["markov28"]}
    for name, (cols, wt) in variants.items():
        raw = walk_forward(df, h, cols, wt)
        preds[name] = pd.concat([online_platt(df, raw, h, W) for W in CAL_WINDOWS], axis=1).mean(axis=1, skipna=False)
        print(f"  h={h} {name} done", flush=True)

    full = df.assign(**preds).dropna(subset=list(variants))
    full = full[(full["active_now"] == 0) & (full["hour_utc"] >= BT_START)].reset_index(drop=True)
    end = full["hour_utc"].max()
    periods = {"до 07.09": (BT_START, REGIME), "з 07.09": (REGIME, end + H),
               "test (28 дн.)": (end + H - pd.Timedelta(days=28), end + H)}
    from sklearn.metrics import roc_auc_score
    rows = []
    for pname, (a, b) in periods.items():
        s = full[(full["hour_utc"] >= a) & (full["hour_utc"] < b)].reset_index(drop=True)
        y = s["target"].to_numpy()
        bm = np.mean((s["markov28"] - y) ** 2)
        for name in ["markov28", *variants]:
            p = s[name].to_numpy()
            r = {"h": h, "period": pname, "model": name, "n": len(s),
                 "BSS_vs_markov": 1 - np.mean((p - y) ** 2) / bm, "AUC": roc_auc_score(y, p),
                 "mean_p": p.mean(), "base_rate": y.mean()}
            if name != "markov28":
                r["dBrier_vs_markov"], r["ci_lo"], r["ci_hi"] = block_bootstrap_diff(s, p, s["markov28"].to_numpy())
                if name != "GT":
                    r["dBrier_vs_GT"], r["ciGT_lo"], r["ciGT_hi"] = block_bootstrap_diff(s, p, s["GT"].to_numpy())
            rows.append(r)
    return rows


def main(hs: list[int]) -> None:
    hourly = pd.read_parquet(PROC_DIR / "hourly_volunteer.parquet")
    intervals = pd.read_parquet(PROC_DIR / "intervals_volunteer.parquet")
    posts = load_posts()
    rows = [r for h in hs for r in run(h, hourly, intervals, posts)]
    res = pd.DataFrame(rows)
    res.to_csv(REPORT_DIR / "regime_fix.csv", index=False)
    show = ["h", "period", "model", "n", "BSS_vs_markov", "AUC", "mean_p", "base_rate",
            "dBrier_vs_markov", "ci_lo", "ci_hi", "dBrier_vs_GT", "ciGT_lo", "ciGT_hi"]
    md = res[show].to_markdown(index=False, floatfmt=".4f")
    (REPORT_DIR / "regime_fix.md").write_text(
        "# Новий режим (з 07.09.2026): відносні ознаки і вага свіжих рядків\n\n"
        "Onset, walk-forward + ансамбль онлайн-Platt. Гіпотезу і кандидата (GT_nw) зафіксовано до запуску.\n"
        "ΔBrier < 0 = модель краща; 95% CI — block bootstrap по днях.\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("h", nargs="*", type=int, default=[1, 3])
    main(p.parse_args().h)
