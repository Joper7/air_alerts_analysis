"""
Крок 4: walk-forward бектест + онлайн-рекалібрування з ансамблем вікон.

1. Walk-forward: кожні 7 днів модель (LightGBM L1.5, offset на markov28 — обрана в
   src.train за val) перенавчається на всіх рядках з t+h ≤ момент перенавчання
   і прогнозує наступні 7 днів. Так отримуємо out-of-sample прогнози p_raw за
   ~8 місяців — і чесний бектест, і матеріал для калібратора.
   Кількість дерев фіксована (не залежить від val/test).
2. Онлайн-Platt: для кожної доби D калібратор logit(p) → a·logit(p_raw) + b
   вчиться на парах (p_raw(s), y(s)) з s ∈ [D − W, D − h], тобто лише на цілях,
   які вже спостережені до D. Вікна W ∈ {7, 14, 28, 56} днів.
3. Ансамбль: середнє відкаліброваних імовірностей за всіма W
   (короткі вікна швидко ловлять зміну режиму, довгі — стабільні).

Вибір фінального варіанта — лише за val (onset log-loss); test — звітність.

Запуск:  python -m src.backtest           # h = 1, 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluate import block_bootstrap_diff, chrono_split, evaluate, reliability_table
from src.features import FEATURE_SETS, make_features
from src.train import LGB_PARAMS, PROC_DIR, REPORT_DIR, _logit, _sigmoid

H = pd.Timedelta(hours=1)
PRED_START = pd.Timestamp("2026-02-01", tz="UTC")   # ≈3 міс. історії для першої моделі
REFIT_EVERY = pd.Timedelta(days=7)
N_TREES = 150
CAL_WINDOWS = (7, 14, 28, 56)
BT_SETS = ("L1.5", "L1.5S", "L1.5G")


# ---------------------------------------------------------------- walk-forward
def walk_forward(df: pd.DataFrame, h: int, cols: list[str]) -> pd.Series:
    """Out-of-sample p_raw для origin ≥ PRED_START; модель бачить лише цілі, відомі до refit."""
    out = pd.Series(np.nan, index=df.index)
    refit = PRED_START
    end = df["hour_utc"].max() + H
    while refit < end:
        tr = df[df["hour_utc"] + h * H <= refit]                      # purge: ціль відома до refit
        te_mask = (df["hour_utc"] >= refit) & (df["hour_utc"] < refit + REFIT_EVERY)
        if te_mask.any():
            ds = lgb.Dataset(tr[cols], tr["target"], init_score=_logit(tr["markov28"].to_numpy()),
                             categorical_feature=["oblast"])
            booster = lgb.train(LGB_PARAMS, ds, num_boost_round=N_TREES)
            te = df[te_mask]
            raw = booster.predict(te[cols], raw_score=True)
            out[te_mask] = _sigmoid(raw + _logit(te["markov28"].to_numpy()))
        refit += REFIT_EVERY
    return out


# ---------------------------------------------------------------- online recalibration
def online_platt(df: pd.DataFrame, p_raw: pd.Series, h: int, window_days: int) -> pd.Series:
    """Platt-калібратор, перенавчений щодоби на останніх `window_days` днях спостережених цілей."""
    out = pd.Series(np.nan, index=df.index)
    x_all = _logit(p_raw.to_numpy())
    t = df["hour_utc"]
    days = pd.date_range(t[p_raw.notna()].min().floor("D"), t.max().ceil("D"), freq="D")
    w = pd.Timedelta(days=window_days)
    for D in days:
        fit = p_raw.notna() & (t >= D - w) & (t + h * H <= D)       # ціль спостережена до D
        if fit.sum() < 24 * window_days * 0.8 * df["oblast"].nunique():   # недостатньо історії
            continue
        pred = p_raw.notna() & (t >= D) & (t < D + pd.Timedelta(days=1))
        if not pred.any():
            continue
        y = df.loc[fit, "target"].to_numpy()
        if y.min() == y.max():
            continue
        lr = LogisticRegression(C=1.0).fit(x_all[fit.to_numpy()].reshape(-1, 1), y)
        out[pred] = lr.predict_proba(x_all[pred.to_numpy()].reshape(-1, 1))[:, 1]
    return out


# ---------------------------------------------------------------- one horizon
def run(hourly, intervals, h: int) -> dict:
    df = make_features(hourly, intervals, h)
    masks, bounds = chrono_split(df, h)

    preds = {"clim": df["clim"], "markov28": df["markov28"]}
    for fs in BT_SETS:
        raw = preds[f"wf_{fs}_raw"] = walk_forward(df, h, FEATURE_SETS[fs])
        cal = {W: online_platt(df, raw, h, W) for W in CAL_WINDOWS}
        for W, p in cal.items():
            preds[f"wf_{fs}_cal{W}"] = p
        preds[f"wf_{fs}_ens"] = pd.concat(cal.values(), axis=1).mean(axis=1, skipna=False)

    res = {}
    for split in ("val", "test"):
        d = df[masks[split]].reset_index(drop=True)
        p = {k: v[masks[split]].to_numpy() for k, v in preds.items()}
        assert not any(np.isnan(v).any() for v in p.values()), f"NaN у прогнозах ({split})"
        res[split] = (d, p, evaluate(d, p).assign(h=h, split=split))

    # вибір за val: найкращий onset log-loss серед wf_*
    vm = res["val"][2]
    cand = vm[(vm["slice"] == "onset") & vm["model"].str.startswith("wf_")]
    sel = cand.sort_values("logloss")["model"].iloc[0]

    d, p, tm = res["test"]
    onset = (d["active_now"] == 0).to_numpy()
    boot = []
    sel_raw = "_".join(sel.split("_")[:2]) + "_raw"
    pairs = [(sel, "markov28"), (sel, "clim"), (sel, sel_raw),
             ("wf_L1.5S_ens", "wf_L1.5_ens"), ("wf_L1.5S_raw", "wf_L1.5_raw")]
    for a, b in pairs:
        if a == b:
            continue
        for sl, m in (("all", None), ("onset", onset)):
            dd, lo, hi = block_bootstrap_diff(d, p[a], p[b], m)
            boot.append({"h": h, "slice": sl, "A": a, "B": b, "dBrier": dd, "ci_lo": lo, "ci_hi": hi})

    # бектест по тижнях: BSS vs markov28 у зрізі onset за весь walk-forward період
    cmp = [f"wf_{fs}_ens" for fs in BT_SETS]
    # лише рядки, де є ВСІ порівнювані прогнози (ансамбль з'являється після 56 днів історії)
    full = df.assign(**{k: v for k, v in preds.items()}).dropna(subset=cmp + [sel]).reset_index(drop=True)

    # bootstrap за ВЕСЬ walk-forward період (≈7 міс., усі прогнози out-of-sample):
    # 4 тижні test замалі, щоб розрізнити ефект ~0.001 Brier. Вибір варіанта робився на val,
    # який входить у цей період, → легкий оптимізм для `sel`.
    on = (full["active_now"] == 0).to_numpy()
    for a, b in [(sel, "markov28"), ("wf_L1.5_ens", "markov28"), ("wf_L1.5S_ens", "wf_L1.5_ens"),
                 ("wf_L1.5G_ens", "wf_L1.5_ens"), ("wf_L1.5G_ens", "markov28")]:
        dd, lo, hi = block_bootstrap_diff(full, full[a].to_numpy(), full[b].to_numpy(), on)
        boot.append({"h": h, "slice": "onset, весь бектест", "A": a, "B": b,
                     "dBrier": dd, "ci_lo": lo, "ci_hi": hi})
    full = full[on]
    week = full["hour_utc"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    se = lambda c: (full[c] - full["target"]) ** 2
    weekly = pd.DataFrame({"week": week, "base_rate": full["target"], "markov28": se("markov28"),
                           **{c: se(c) for c in cmp}}).groupby("week").mean()
    for c in cmp:
        weekly[f"bss_{c}"] = 1 - weekly[c] / weekly["markov28"]

    rel = reliability_table(d["target"].to_numpy()[onset], p[sel][onset]).assign(h=h, model=sel)
    test_preds = d[["oblast", "hour_utc", "active_now", "target"]].assign(**p)
    return dict(val=vm.assign(selected=lambda x: x["model"] == sel),
                test=tm.assign(selected=lambda x: x["model"] == sel),
                boot=pd.DataFrame(boot), weekly=weekly, rel=rel, sel=sel,
                preds=test_preds, bounds=bounds)


def _table(m: pd.DataFrame) -> str:
    cols = ["model", "brier", "logloss", "bss_clim", "bss_markov", "ece", "selected"]
    out = []
    for sl in ("all", "onset"):
        d = m[m["slice"] == sl]
        out.append(f"\n**{sl}** (n={d['n'].iloc[0]}, base rate={d['base_rate'].iloc[0]:.3f})\n")
        out.append(d[cols].to_markdown(index=False, floatfmt=".4f"))
    return "\n".join(out)


def main(hs: list[int]) -> None:
    hourly = pd.read_parquet(PROC_DIR / "hourly_volunteer.parquet")
    intervals = pd.read_parquet(PROC_DIR / "intervals_volunteer.parquet")
    meta = json.loads((PROC_DIR / "meta_volunteer.json").read_text())
    lines = ["# Walk-forward бектест + онлайн-рекалібрування",
             f"\nЗнімок {meta['snapshot_utc']}. Перенавчання моделі кожні 7 днів з {PRED_START:%Y-%m-%d}; "
             f"Platt щодоби на вікнах {CAL_WINDOWS} днів; `wf_<набір>_ens` = середнє за вікнами. "
             "Вибір — за val onset log-loss."]
    for h in hs:
        print(f"--- h={h}")
        r = run(hourly, intervals, h)
        print("selected:", r["sel"])
        print(r["test"].pivot_table(index="model", columns="slice", values="brier").round(4))
        sfx = f"h{h}"
        r["test"].to_csv(REPORT_DIR / f"backtest_test_{sfx}.csv", index=False)
        r["val"].to_csv(REPORT_DIR / f"backtest_val_{sfx}.csv", index=False)
        r["weekly"].to_csv(REPORT_DIR / f"backtest_weekly_{sfx}.csv")
        r["rel"].to_csv(REPORT_DIR / f"backtest_reliability_{sfx}.csv", index=False)
        r["preds"].to_parquet(REPORT_DIR / f"backtest_preds_{sfx}.parquet", index=False)
        b = r["bounds"]
        lines += [f"\n## h = {h} — обрано за val: `{r['sel']}`",
                  f"val {b['val_start']:%Y-%m-%d} → {b['test_start']:%Y-%m-%d}, "
                  f"test {b['test_start']:%Y-%m-%d} → {b['end']:%Y-%m-%d}",
                  "\n### Test", _table(r["test"]),
                  "\n### Val", _table(r["val"]),
                  "\n### ΔBrier на test, 95% CI (block bootstrap; < 0 = A краща)\n",
                  r["boot"][["slice", "A", "B", "dBrier", "ci_lo", "ci_hi"]].to_markdown(index=False, floatfmt=".5f"),
                  "\n### Бектест по тижнях (onset): BSS відносно markov28\n",
                  r["weekly"][["base_rate"] + [f"bss_wf_{fs}_ens" for fs in BT_SETS]].to_markdown(floatfmt=".3f"),
                  "\n### Reliability обраної моделі на test (onset)\n",
                  r["rel"][["bin", "p_mean", "y_rate", "n"]].to_markdown(index=False, floatfmt=".3f")]
    (REPORT_DIR / "backtest_summary.md").write_text("\n".join(lines) + "\n")
    print(f"-> {REPORT_DIR}/backtest_summary.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("h", nargs="*", type=int, default=[1, 3])
    main(p.parse_args().h)
