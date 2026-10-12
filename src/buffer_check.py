"""
Перевірка стійкості Telegram-ознак до буфера: пост враховується, лише якщо available_at < t − buffer.
Мотивація: у ~25% тривог Києва/Київщини volunteer фіксує старт на >2 хв пізніше, ніж official
(p90 ≈ 10–13 хв), тож пост «за 6 хв до» запису volunteer міг бути вже після реального старту.
Буфер 20/30 хв відсікає цей артефакт.

Запуск: python -m src.buffer_check  (~6 хв)
"""
import pandas as pd
import src.text_features as tf
from src.features import make_features, FEATURE_SETS
from src.backtest import walk_forward, online_platt, CAL_WINDOWS
from src.train import load_posts
from src.evaluate import block_bootstrap_diff
hourly = pd.read_parquet("data/processed/hourly_volunteer.parquet"); iv = pd.read_parquet("data/processed/intervals_volunteer.parquet")
posts = load_posts()
def ens(df, h, cols):
    raw = walk_forward(df, h, cols)
    return pd.concat([online_platt(df, raw, h, W) for W in CAL_WINDOWS], axis=1).mean(axis=1, skipna=False)
for h in (1, 3):
    res = {}
    for buf in (5, 20, 30):
        tf.BUFFER = pd.Timedelta(minutes=buf)
        df = make_features(hourly, iv, h, "window", posts=posts)
        if buf == 5:
            res["G"] = ens(df, h, FEATURE_SETS["L1.5G"])
        res[f"GT_buf{buf}"] = ens(df, h, FEATURE_SETS["L1.5GT"])
    ok = pd.concat(res, axis=1).notna().all(axis=1) & (df["active_now"] == 0)
    d = df[ok].reset_index(drop=True); ref = ((d.markov28 - d.target) ** 2).mean()
    for k, p in res.items():
        dd, lo, hi = block_bootstrap_diff(d, p[ok].to_numpy(), d.markov28.to_numpy())
        line = f"h={h} {k:9s} BSS vs Markov {-dd/ref:+.3f} [{-hi/ref:+.3f}; {-lo/ref:+.3f}]"
        if k != "G":
            dd2, lo2, hi2 = block_bootstrap_diff(d, p[ok].to_numpy(), res["G"][ok].to_numpy())
            line += f" | vs G ΔBrier {dd2:+.5f} [{lo2:+.5f}; {hi2:+.5f}]"
        print(line, flush=True)
