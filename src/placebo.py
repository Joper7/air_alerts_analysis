"""
Перевірки на перенавчання для набору L1.5GT (h=1, 3; зріз onset):
  1) плацебо — пости зсунуті на ±7 діб: справжній сигнал має зникнути, а «гучність каналу як
     мітка періоду» лишилась би;
  2) Brier на навчальних даних vs на нових (walk-forward) vs Markov.
Запуск: python -m src.placebo   (~5 хв)
"""

import numpy as np, pandas as pd, lightgbm as lgb
from src.features import make_features, FEATURE_SETS
from src.backtest import walk_forward, online_platt, CAL_WINDOWS, N_TREES
from src.train import load_posts, LGB_PARAMS, _logit, _sigmoid
from src.evaluate import block_bootstrap_diff

hourly = pd.read_parquet("data/processed/hourly_volunteer.parquet")
iv = pd.read_parquet("data/processed/intervals_volunteer.parquet")
posts0 = load_posts()
cols = FEATURE_SETS["L1.5GT"]

def ens(df, h):
    raw = walk_forward(df, h, cols)
    cal = pd.concat([online_platt(df, raw, h, W) for W in CAL_WINDOWS], axis=1).mean(axis=1, skipna=False)
    return raw, cal

def score(df, p):
    m = p.notna() & (df["active_now"] == 0)
    d = df[m].reset_index(drop=True); pp = p[m].to_numpy()
    ref = ((d["markov28"] - d["target"]) ** 2).mean()
    dd, lo, hi = block_bootstrap_diff(d, pp, d["markov28"].to_numpy())
    return f"BSS {-dd/ref:+.3f} [{-hi/ref:+.3f}; {-lo/ref:+.3f}]"

for h in (1, 3):
    for name, shift in [("реальні пости", None), ("плацебо: пости +7 діб", pd.Timedelta(days=7)),
                        ("плацебо: пости −7 діб", pd.Timedelta(days=-7))]:
        posts = posts0.copy()
        if shift is not None:
            for c in ("date_utc", "edit_date_utc", "available_at"):
                posts[c] = posts[c] + shift
        df = make_features(hourly, iv, h, "window", posts=posts)
        raw, cal = ens(df, h)
        print(f"h={h} {name:24s} {score(df, cal)}", flush=True)
    # in-sample vs out-of-sample (реальні пости)
    df = make_features(hourly, iv, h, "window", posts=posts0)
    b = lgb.train(LGB_PARAMS, lgb.Dataset(df[cols], df["target"], init_score=_logit(df["markov28"].to_numpy()),
                                          categorical_feature=["oblast"]), N_TREES)
    p_in = pd.Series(_sigmoid(b.predict(df[cols], raw_score=True) + _logit(df["markov28"].to_numpy())), index=df.index)
    raw, _ = ens(df, h)
    m = raw.notna() & (df["active_now"] == 0)
    br = lambda p: ((p[m] - df.loc[m, "target"]) ** 2).mean()
    print(f"h={h} Brier на навчальних даних {br(p_in):.4f} | на нових (walk-forward, без калібрування) {br(raw):.4f} | Markov {br(df['markov28']):.4f}", flush=True)
