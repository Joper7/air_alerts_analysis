"""
Графіки для README → reports/figures/*.png

  drift.png        — частка годин під тривогою по місяцях (4 області скоупу)
  skill_curve.png  — BSS(h) відносно онлайн-Markov на всьому walk-forward (onset), 95% CI
  weekly_bss.png   — BSS по тижнях бектесту (фінальна модель, h=1)
  reliability.png  — діаграма надійності фінальної моделі на test (onset)

Запуск (після src.backtest для всіх h):  python -m src.plots
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import block_bootstrap_diff
from src.features import SCOPE

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
HS = [1, 3, 6, 12, 24]
# Референсна палітра (dataviz skill), світлий режим: категоріальні слоти 1–4 у фіксованому порядку
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"wf_L1.5_ens": "історія тривог (L1.5)",
          "wf_L1.5G_ens": "+ стан усіх областей (G)",
          "wf_L1.5GT_ens": "+ Telegram @kpszsu (T)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("->", FIG_DIR / name)


def drift():
    h = pd.read_parquet("data/processed/hourly_volunteer.parquet")
    h = h[h["oblast"].isin(SCOPE) & (h["hour_utc"] >= "2025-09-01")]
    h = h.assign(month=h["hour_utc"].dt.tz_convert(None).dt.to_period("M"),
                 ts=h["hour_utc"].dt.tz_convert(None))
    m = h.groupby(["month", "oblast"])["alert"].mean().unstack()
    m.index = h.groupby("month")["ts"].mean().loc[m.index]          # точка = середина наявних днів місяця
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for c, ob in zip(SERIES, SCOPE):
        ax.plot(m.index, m[ob], color=c, lw=2, marker="o", ms=4, label=ob)
        ax.annotate(ob.replace(" oblast", ""), (m.index[-1], m[ob].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.axvspan(pd.Timestamp("2026-09-12"), pd.Timestamp("2026-10-10"), color=GRID, alpha=0.6, lw=0)
    ax.set_ylim(bottom=0)
    ax.text(pd.Timestamp("2026-09-14"), 0.01, "test", color=INK2, va="bottom", fontsize=9)
    ax.set_ylabel("частка годин під тривогою")
    ax.set_title("Дрейф: рівень тривог у скоупі подвоївся у вересні 2026 (жовтень — 10 днів)")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "drift.png")


def skill_curve(models: list[str]):
    rows = []
    for h in HS:
        path = REPORT_DIR / f"backtest_wf_h{h}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        d = d[d["active_now"] == 0].reset_index(drop=True)
        ref = ((d["markov28"] - d["target"]) ** 2).mean()
        for m in models + ["clim"]:
            if m not in d:
                continue
            dd, lo, hi = block_bootstrap_diff(d, d[m].to_numpy(), d["markov28"].to_numpy())
            rows.append({"h": h, "model": m, "bss": -dd / ref, "lo": -hi / ref, "hi": -lo / ref})
    r = pd.DataFrame(rows)
    r.to_csv(REPORT_DIR / "skill_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color=INK2, lw=1)
    ax.text(HS[-1], 0.002, "онлайн-Markov (бейзлайн)", ha="right", va="bottom", color=INK2, fontsize=9)
    present = [m for m in models + ["clim"] if m in set(r["model"])]
    for i, m in enumerate(present):
        x = r[r["model"] == m]
        c = SERIES[i] if m != "clim" else "#8a8984"
        lab = LABELS.get(m, "кліматологія")
        ax.errorbar(x["h"], x["bss"], yerr=[x["bss"] - x["lo"], x["hi"] - x["bss"]], color=c, lw=2,
                    marker="o", ms=5, capsize=3, label=lab, ls="--" if m == "clim" else "-")
    ax.set_xscale("log")
    ax.set_xticks(HS, [str(h) for h in HS])
    ax.minorticks_off()
    ax.set_xlabel("горизонт h, год")
    ax.set_ylabel("BSS відносно онлайн-Markov")
    ax.set_title("Якість прогнозу початку тривоги за горизонтом (walk-forward, 95% CI)")
    ax.legend(fontsize=9, loc="best")
    _save(fig, "skill_curve.png")
    return r


def weekly(final: str, h: int = 1):
    d = pd.read_parquet(REPORT_DIR / f"backtest_wf_h{h}.parquet")
    d = d[d["active_now"] == 0]
    wk = d["hour_utc"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    se = lambda c: (d[c] - d["target"]) ** 2
    w = pd.DataFrame({"wk": wk, "m": se(final), "b": se("markov28")}).groupby("wk").mean()
    bss = 1 - w["m"] / w["b"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(bss.index, bss, width=5, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=INK2, lw=1)
    ax.axvline(pd.Timestamp("2026-09-12"), color=INK2, lw=1, ls=":")
    ax.text(pd.Timestamp("2026-09-13"), ax.get_ylim()[1] * 0.95, "test →", color=INK2, va="top", fontsize=9)
    ax.set_ylabel("BSS відносно онлайн-Markov")
    ax.set_title(f"Бектест по тижнях: {LABELS.get(final, final)}, h={h}, onset "
                 f"(краще в {(bss > 0).sum()} з {len(bss)} тижнів)")
    _save(fig, "weekly_bss.png")


def reliability(final: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for c, h in zip(SERIES, (1, 3)):
        path = REPORT_DIR / f"backtest_preds_h{h}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path)
        d = d[d["active_now"] == 0]
        b = np.clip((d[final] * 10).astype(int), 0, 9)
        t = d.groupby(b).agg(p=(final, "mean"), y=("target", "mean"), n=("target", "size"))
        t = t[t["n"] >= 30]
        ax.plot(t["p"], t["y"], color=c, lw=2, marker="o", ms=5, label=f"h={h}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("прогноз моделі"); ax.set_ylabel("фактична частота")
    ax.set_title("Надійність на test (onset)")
    ax.legend(loc="upper left")
    _save(fig, "reliability.png")


def main():
    wf1 = pd.read_parquet(REPORT_DIR / "backtest_wf_h1.parquet")
    models = [m for m in LABELS if m in wf1]
    final = models[-1]
    drift()
    skill_curve(models)
    weekly(final)
    reliability(final)


if __name__ == "__main__":
    main()
