"""
Експеримент на OFFICIAL-датасеті (окремо від volunteer — джерела не склеюємо).

Official після реформи містить тривоги ПО РАЙОНАХ (Київщина — 7, Черкащина і Кіровоградщина — по 4;
Київ — міський рівень). Ціль та сама: «будь-яка тривога в області» (union районів).
Нові ознаки рівня загрози (набір LV), усі — стан на момент t або старти в [t−1 год, t):
  lv_share_self      частка районів своєї області під тривогою зараз
  lv_share_self_1h   те саме годину тому
  lv_starts_self_1h  к-сть районів своєї області, що увімкнули тривогу за останню годину
  lv_share_nb        середня частка районів під тривогою в сусідніх областях
  lv_starts_nb_1h    к-сть стартів у районах сусідніх областей за годину
  lv_units_nat       к-сть районів під тривогою по країні
  lv_starts_nat_1h   к-сть стартів районних тривог по країні за годину
Порівняння walk-forward (те саме, що src/backtest.py): L1.5G vs L1.5G+LV, L1.5GT vs L1.5GT+LV.
Official закінчується 07.09.2026 → оцінка на 02–09.2026, а не на найсвіжішому періоді.

Запуск (після `python -m src.data_loader official`):  python -m src.official_levels → reports/official_levels.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import CAL_WINDOWS, online_platt, walk_forward
from src.data_loader import (PERMANENT_ALERT_REGIONS, SOURCE_FILES, _drop_bad_intervals, _parse_times,
                             download, merge_intervals)
from src.evaluate import block_bootstrap_diff
from src.features import FEATURE_SETS, NEIGHBORS, SCOPE, make_features
from src.train import load_posts

PROC = Path("data/processed")
_NS_H = 3_600_000_000_000
LV_COLS = ["lv_share_self", "lv_share_self_1h", "lv_starts_self_1h", "lv_share_nb", "lv_starts_nb_1h",
           "lv_units_nat", "lv_starts_nat_1h"]


def unit_intervals(since: str = "2025-09-01") -> pd.DataFrame:
    """Інтервали на рівні «одиниці» = район (районні й громадські тривоги → свій район; обласні → уся область)."""
    df = _parse_times(pd.read_csv(download(SOURCE_FILES["official"])))
    df = df[~df["oblast"].isin(PERMANENT_ALERT_REGIONS) & (df["started_at"] >= pd.Timestamp(since, tz="UTC"))]
    df = _drop_bad_intervals(df.drop_duplicates())
    df["unit"] = np.where(df["raion"].notna(), df["oblast"] + "|" + df["raion"].astype(str), df["oblast"] + "|*")
    return df


def level_features(df_units: pd.DataFrame, origins: pd.DatetimeIndex) -> tuple[dict, dict]:
    """Для кожної одиниці: активна в момент t і к-сть стартів у [t−1 год, t). Обласна тривога («|*») = усі райони."""
    T = origins.as_unit("ns").asi8
    units_of = df_units.groupby("oblast")["unit"].unique().to_dict()
    raions_of = {ob: [u for u in us if not u.endswith("|*")] or [ob + "|*"] for ob, us in units_of.items()}
    m = merge_intervals(df_units, key="unit")
    act, st1 = {}, {}
    for u, g in m.groupby("unit"):
        s = pd.DatetimeIndex(g["started_at"]).as_unit("ns").asi8
        e = pd.DatetimeIndex(g["finished_at"]).as_unit("ns").asi8
        nb = np.searchsorted(s, T, side="left")
        j = np.clip(nb - 1, 0, None)
        act[u] = ((nb > 0) & (e[j] > T)).astype(float)
        st1[u] = (nb - np.searchsorted(s, T - _NS_H, side="left")).astype(float)
    share, starts = {}, {}
    for ob, rs in raions_of.items():
        whole = act.get(ob + "|*", np.zeros(len(T)))                  # обласна тривога покриває всі райони
        a = np.mean([np.maximum(act.get(r, 0.0), whole) for r in rs], axis=0)
        share[ob] = pd.Series(a, index=origins)
        starts[ob] = pd.Series(np.sum([st1.get(r, 0.0) for r in rs], axis=0)
                               + st1.get(ob + "|*", 0.0), index=origins)
    return share, starts


def add_level_features(df: pd.DataFrame, share: dict, starts: dict) -> pd.DataFrame:
    origins = pd.DatetimeIndex(sorted(df["hour_utc"].unique()))
    nat_units = sum(share.values())                                  # сума часток = «областей-еквівалентів» під тривогою
    nat_starts = sum(starts.values())
    parts = []
    for ob in SCOPE:
        nb = [n for n in NEIGHBORS[ob] if n in share]
        p = pd.DataFrame({
            "hour_utc": origins,
            "oblast": ob,
            "lv_share_self": share[ob].reindex(origins).to_numpy(),
            "lv_share_self_1h": share[ob].shift(1).reindex(origins).to_numpy(),
            "lv_starts_self_1h": starts[ob].reindex(origins).to_numpy(),
            "lv_share_nb": pd.concat([share[n] for n in nb], axis=1).mean(axis=1).reindex(origins).to_numpy(),
            "lv_starts_nb_1h": sum(starts[n] for n in nb).reindex(origins).to_numpy(),
            "lv_units_nat": nat_units.reindex(origins).to_numpy(),
            "lv_starts_nat_1h": nat_starts.reindex(origins).to_numpy(),
        })
        parts.append(p)
    lv = pd.concat(parts, ignore_index=True)
    lv["oblast"] = pd.Categorical(lv["oblast"], categories=df["oblast"].cat.categories)
    return df.merge(lv, on=["oblast", "hour_utc"], how="left")


def ens(df: pd.DataFrame, h: int, cols: list[str]) -> pd.Series:
    raw = walk_forward(df, h, cols)
    return pd.concat([online_platt(df, raw, h, W) for W in CAL_WINDOWS], axis=1).mean(axis=1, skipna=False)


def main(hs=(1, 3)) -> None:
    hourly = pd.read_parquet(PROC / "hourly_official.parquet")
    intervals = pd.read_parquet(PROC / "intervals_official.parquet")
    posts = load_posts()
    units = unit_intervals()
    lines = ["# Official-датасет: рівні загрози по районах (набір LV)", "",
             f"Одиниць (районів + міст): {units['unit'].nunique()}; сітка до {hourly['hour_utc'].max():%Y-%m-%d}. "
             "Walk-forward з 01.02.2026, рекалібрування 7/14/28/56 днів. Зріз onset = в області тривоги зараз немає.", ""]
    for h in hs:
        df = make_features(hourly, intervals, h, "window", posts=posts)
        origins = pd.DatetimeIndex(sorted(df["hour_utc"].unique()))
        share, starts = level_features(units, origins)
        df = add_level_features(df, share, starts)
        base = {"L1.5G": FEATURE_SETS["L1.5G"], "L1.5GT": FEATURE_SETS["L1.5GT"]}
        preds = {}
        for nm, cols in base.items():
            preds[nm] = ens(df, h, cols)
            preds[nm + "+LV"] = ens(df, h, cols + LV_COLS)
        ok = pd.concat(preds, axis=1).notna().all(axis=1)
        lines += [f"## h = {h}", "", "| модель | зріз | BSS vs Markov | 95% CI |", "|---|---|---|---|"]
        for sl, m in (("onset", ok & (df["active_now"] == 0)), ("all", ok)):
            d = df[m].reset_index(drop=True)
            ref = ((d["markov28"] - d["target"]) ** 2).mean()
            for nm, p in preds.items():
                dd, lo, hi = block_bootstrap_diff(d, p[m].to_numpy(), d["markov28"].to_numpy())
                lines.append(f"| {nm} | {sl} | {-dd / ref:+.3f} | [{-hi / ref:+.3f}; {-lo / ref:+.3f}] |")
        lines += ["", "ΔBrier від додавання LV (< 0 = LV допомагає):", ""]
        for sl, m in (("onset", ok & (df["active_now"] == 0)), ("all", ok)):
            d = df[m].reset_index(drop=True)
            for nm in base:
                dd, lo, hi = block_bootstrap_diff(d, preds[nm + "+LV"][m].to_numpy(), preds[nm][m].to_numpy())
                lines.append(f"- {nm}, {sl}: {dd:+.5f} [{lo:+.5f}; {hi:+.5f}]")
        lines.append("")
        print("\n".join(lines[-20:]), flush=True)
    Path("reports/official_levels.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
