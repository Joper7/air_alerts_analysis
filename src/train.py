"""
Крок 3: бейзлайни + LightGBM (абляція L1 → L1.5) + LogReg-контроль для задачі A.

Протокол:
  1. fit на train, early stopping по log-loss на val → best_iter.
  2. фінальна модель = refit на train+val з best_iter (дрейф режиму: свіжі 4 тижні
     важливіші за «чистий» val). Статичний Markov теж перефічується на train+val.
  3. ВИБІР фінального варіанта — лише за val log-loss у зрізі onset (`selected` у звіті).
     Test — тільки звітність.

Варіанти LightGBM (через дрейф: рівень тривог у 09–10.2026 ≈ 2× від train):
  plain — абсолютна ймовірність;
  off   — offset: logit(p) = logit(markov28) + f(x); дерева вчать ПОПРАВКУ до
          онлайн-бейзлайну, який сам підлаштовується під рівень (дерева не екстраполюють);
  off_w — offset + вага свіжих рядків (half-life 60 днів).

Запуск:  python -m src.train            # h = 1, 3
         python -m src.train 1 3 6 12 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import (block_bootstrap_diff, chrono_split, evaluate, markov_fit,
                          markov_predict, reliability_table)
from src.features import FEATURE_SETS, make_features

PROC_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42, num_threads=4,
)
CYCLIC = ["hour_t", "dow_t", "hour_target", "dow_target"]


HALF_LIFE_DAYS = 60
VARIANTS = {"plain": dict(offset=False, weight=False),
            "off": dict(offset=True, weight=False),
            "off_w": dict(offset=True, weight=True)}


# ---------------------------------------------------------------- models
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _dataset(d: pd.DataFrame, cols, offset: bool, weight: bool, reference=None):
    w = None
    if weight:   # експоненційна вага за віком відносно кінця цього набору
        age = (d["hour_utc"].max() - d["hour_utc"]).dt.total_seconds() / 86400
        w = 0.5 ** (age / HALF_LIFE_DAYS).to_numpy()
    init = _logit(d["markov28"].to_numpy()) if offset else None
    return lgb.Dataset(d[cols], d["target"], weight=w, init_score=init,
                       categorical_feature=["oblast"], reference=reference)


def lgb_predict(booster, d: pd.DataFrame, cols, offset: bool, num_iteration=None):
    if not offset:
        return booster.predict(d[cols], num_iteration=num_iteration)
    raw = booster.predict(d[cols], raw_score=True, num_iteration=num_iteration)
    return _sigmoid(raw + _logit(d["markov28"].to_numpy()))


def fit_lgb(tr, va, cols, offset, weight):
    dtr = _dataset(tr, cols, offset, weight)
    dva = _dataset(va, cols, offset, False, reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster, max(booster.best_iteration, 1)


def refit_lgb(d, cols, offset, weight, n_iter):
    return lgb.train(LGB_PARAMS, _dataset(d, cols, offset, weight), num_boost_round=n_iter)


def _cyclic(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c, period in zip(CYCLIC, (24, 7, 24, 7)):
        X[c + "_sin"] = np.sin(2 * np.pi * X[c] / period)
        X[c + "_cos"] = np.cos(2 * np.pi * X[c] / period)
    return X.drop(columns=CYCLIC)


def make_logreg(cols: list[str]):
    num = [c for c in _cyclic(pd.DataFrame(columns=cols)).columns if c != "oblast"]
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), ["oblast"]),
                             ("sc", StandardScaler(), num)])        # scaler fit лише на train
    return make_pipeline(pre, LogisticRegression(C=0.3, max_iter=3000))


# ---------------------------------------------------------------- one horizon
def run_horizon(hourly, intervals, h: int, kind: str = "window") -> dict:
    df = make_features(hourly, intervals, h, kind)
    masks, bounds = chrono_split(df, h)
    tr, va, te = (df[masks[k]].reset_index(drop=True) for k in ("train", "val", "test"))
    trva = pd.concat([tr, va], ignore_index=True)

    val_preds = {k: va[k].to_numpy() for k in ("clim", "markov28")}
    test_preds = {k: te[k].to_numpy() for k in ("clim", "markov28")}

    # Статичний Markov (таблиця з train) — як у початковому плані
    val_preds["markov_static"] = markov_predict(markov_fit(tr), va, tr["target"].mean())
    test_preds["markov_static"] = markov_predict(markov_fit(trva), te, trva["target"].mean())

    # LightGBM: абляція L1 → L1.5 × варіанти
    importance, best_iters = {}, {}
    for fs, cols in FEATURE_SETS.items():
        for vname, v in VARIANTS.items():
            name = f"lgb_{fs}_{vname}"
            booster, best = fit_lgb(tr, va, cols, **v)
            val_preds[name] = lgb_predict(booster, va, cols, v["offset"], best)
            final = refit_lgb(trva, cols, v["offset"], v["weight"], best)
            test_preds[name] = lgb_predict(final, te, cols, v["offset"])
            importance[name] = pd.Series(final.feature_importance("gain"), index=cols)
            best_iters[name] = best

    # LogReg-контроль на L1.5
    cols = FEATURE_SETS["L1.5"]
    lr = make_logreg(cols).fit(_cyclic(tr[cols]), tr["target"])
    val_preds["logreg_L1.5"] = lr.predict_proba(_cyclic(va[cols]))[:, 1]
    lr = make_logreg(cols).fit(_cyclic(trva[cols]), trva["target"])
    test_preds["logreg_L1.5"] = lr.predict_proba(_cyclic(te[cols]))[:, 1]

    val_m = evaluate(va, val_preds).assign(h=h, kind=kind)
    test_m = evaluate(te, test_preds).assign(h=h, kind=kind)

    # Вибір фінального варіанта ЛИШЕ за val (onset log-loss) серед LightGBM на L1.5
    cand = val_m[(val_m["slice"] == "onset") & val_m["model"].str.startswith("lgb_L1.5")]
    sel = cand.sort_values("logloss")["model"].iloc[0]
    variant = "_".join(sel.split("_")[2:])                 # абляція: той самий варіант на інших наборах
    ablation = [f"lgb_{fs}_{variant}" for fs in FEATURE_SETS]
    for m in (val_m, test_m):
        m["selected"] = m["model"] == sel

    onset = (te["active_now"] == 0).to_numpy()
    boot = []
    pairs = [(sel, "markov28"), (sel, "clim"), (sel, "logreg_L1.5")] + [(sel, a) for a in ablation]
    for a, b in pairs:
        if a == b:
            continue
        for sl, m in (("all", None), ("onset", onset)):
            d, lo, hi = block_bootstrap_diff(te, test_preds[a], test_preds[b], m)
            boot.append({"h": h, "kind": kind, "slice": sl, "A": a, "B": b,
                         "dBrier": d, "ci_lo": lo, "ci_hi": hi})

    rel = reliability_table(te["target"].to_numpy(), test_preds[sel]).assign(h=h, model=sel)
    info = {
        "h": h, "kind": kind,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "train": [str(tr.hour_utc.min()), str(tr.hour_utc.max())],
        "val": [str(va.hour_utc.min()), str(va.hour_utc.max())],
        "test": [str(te.hour_utc.min()), str(te.hour_utc.max())],
        "best_iter": best_iters, "selected": sel,
    }
    imp = importance[sel].sort_values(ascending=False)
    preds = te[["oblast", "hour_utc", "active_now", "target"]].assign(**test_preds)
    return dict(val=val_m, test=test_m, boot=pd.DataFrame(boot), rel=rel, info=info, imp=imp,
                preds=preds)


# ---------------------------------------------------------------- report
def _fmt_table(m: pd.DataFrame) -> str:
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
    REPORT_DIR.mkdir(exist_ok=True)

    res = {}
    for h in hs:
        kind = "window" if h <= 3 else "point"
        print(f"--- h={h} ({kind})")
        res[h] = r = run_horizon(hourly, intervals, h, kind)
        print(json.dumps(r["info"], ensure_ascii=False))
        print(r["test"].pivot_table(index="model", columns="slice", values="brier").round(4))
        print(r["val"].pivot_table(index="model", columns="slice", values="logloss").round(4))

    val = pd.concat([r["val"] for r in res.values()])
    test = pd.concat([r["test"] for r in res.values()])
    boot = pd.concat([r["boot"] for r in res.values()])
    rel = pd.concat([r["rel"] for r in res.values()])
    suffix = "_".join(map(str, hs))
    val.to_csv(REPORT_DIR / f"val_metrics_h{suffix}.csv", index=False)
    test.to_csv(REPORT_DIR / f"test_metrics_h{suffix}.csv", index=False)
    boot.to_csv(REPORT_DIR / f"bootstrap_h{suffix}.csv", index=False)
    rel.to_csv(REPORT_DIR / f"reliability_h{suffix}.csv", index=False)
    for h, r in res.items():   # прогнози на test — для графіків і бектесту
        r["preds"].to_parquet(REPORT_DIR / f"test_preds_h{h}.parquet", index=False)

    lines = [f"# Результати задачі A (h = {', '.join(map(str, hs))})",
             f"\nДжерело: volunteer, знімок {meta['snapshot_utc']}, сітка до {meta['data_cutoff_utc']}."]
    for h, r in res.items():
        i = r["info"]
        lines += [f"\n## h = {h} ({i['kind']})",
                  f"train {i['train'][0][:10]} → {i['train'][1][:10]} (n={i['n_train']}), "
                  f"val {i['val'][0][:10]} → {i['val'][1][:10]} (n={i['n_val']}), "
                  f"test {i['test'][0][:10]} → {i['test'][1][:10]} (n={i['n_test']}); "
                  f"**обрано за val: `{i['selected']}`**",
                  "\n### Test", _fmt_table(r["test"]),
                  "\n### Val (модель на train, для рішення про калібрування)", _fmt_table(r["val"]),
                  "\n### ΔBrier, 95% CI (block bootstrap по днях; < 0 = A краща)\n",
                  r["boot"][["slice", "A", "B", "dBrier", "ci_lo", "ci_hi"]]
                  .to_markdown(index=False, floatfmt=".5f"),
                  f"\n### Топ-10 ознак {i['selected']} (gain)\n",
                  (r["imp"].head(10) / r["imp"].sum()).round(3).to_frame("gain_share").to_markdown()]
    (REPORT_DIR / f"summary_h{suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"-> {REPORT_DIR}/summary_h{suffix}.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("h", nargs="*", type=int, default=[1, 3])
    main(p.parse_args().h)
