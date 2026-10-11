"""
Крок 2: ознаки L1 / L1.5 і цілі для задачі A.

Конвенції часу (усе в UTC):
  - origin t — межа години, момент прогнозу; рядок датасету індексується t (`hour_utc`).
  - рядок погодинної сітки r описує інтервал [r, r+1h).
  - ознаки в момент t бачать лише рядки сітки r < t (тобто r ≤ t−1h)
    + стан на момент t з інтервалів (чи триває тривога зараз, скільки триває) —
    це відомо в t і не потребує жодних даних після t.
  - ціль window(h) = будь-яка тривога в рядках t..t+h−1, тобто у (t, t+h];
    ціль point(h)  = тривога в рядку t+h−1 (остання година вікна).

Перевірка відсутності leakage: tests/test_features.py (обрізання даних у t0 не
змінює жодної ознаки для origin ≤ t0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import KYIV_TZ, REFORM_END

SCOPE = ["Kyiv City", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast"]

# Сусіди «вгору за маршрутом» (зафіксовано в CLAUDE.md)
NEIGHBORS = {
    "Kyiv City": ["Kyivska oblast", "Chernihivska oblast", "Sumska oblast",
                  "Poltavska oblast", "Zhytomyrska oblast"],
    "Kyivska oblast": ["Kyiv City", "Chernihivska oblast", "Sumska oblast",
                       "Poltavska oblast", "Zhytomyrska oblast"],
    "Cherkaska oblast": ["Poltavska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast",
                         "Odeska oblast", "Kirovohradska oblast"],
    "Kirovohradska oblast": ["Poltavska oblast", "Dnipropetrovska oblast", "Mykolaivska oblast",
                             "Odeska oblast", "Cherkaska oblast"],
}

MASS_THRESHOLD = 16          # ≥16 областей з тривогою в одну годину = масована атака
CLIM_DAYS = 28
HISTORY_DAYS = 60            # запас історії до REFORM_END для 14-денних вікон і кліматології
H = pd.Timedelta(hours=1)
_NS_H = 3_600_000_000_000
_NS_MIN = 60_000_000_000

LAGS = list(range(1, 25))
L1_COLS = (
    ["active_now", "elapsed_min", "log_since_end_min", "starts_24h", "starts_7d"]
    + [f"lag_{k}" for k in LAGS]
    + ["min_lag_1", "min_lag_2", "min_lag_3"]
    + ["roll_6", "roll_24", "roll_168", "roll_336"]
    + ["clim", "markov28", "hour_t", "dow_t", "hour_target", "dow_target", "oblast"]
)
L15_COLS = [
    "nb_active_n", "nb_starts_1h", "nb_starts_3h", "nb_alert_3h",
    "n_active_now", "d_n_active", "n_alert_lasthour", "n_alert_max_6h",
    "hours_since_mass", "mass_count_14d",
]
FEATURE_SETS = {"L1": L1_COLS, "L1.5": L1_COLS + L15_COLS}


# ---------------------------------------------------------------- стан з інтервалів
def interval_state(intervals: pd.DataFrame, origins: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """
    Стан кожної області на момент t (відомий у t):
      active      — тривога триває в момент t (старт < t, відбій > t)
      elapsed     — хвилин від початку поточної тривоги (0, якщо немає)
      since_end   — хвилин від останнього відбою (0, якщо тривога триває)
      starts_<w>h — кількість стартів тривог у [t−w, t)
    Інтервали вже злиті (merge_intervals), тому блоки не перекриваються.
    """
    T = origins.as_unit("ns").asi8
    cols: dict[str, dict[str, np.ndarray]] = {}
    for ob, g in intervals.groupby("oblast"):
        s = pd.DatetimeIndex(g["started_at"]).as_unit("ns").asi8
        e = pd.DatetimeIndex(g["finished_at"]).as_unit("ns").asi8
        order = np.argsort(s)
        s, e = s[order], e[order]

        n_before = np.searchsorted(s, T, side="left")       # к-сть стартів < t
        j = np.clip(n_before - 1, 0, None)                  # останній блок зі стартом < t
        has = n_before > 0
        active = has & (e[j] > T)
        res = {
            "active": active.astype(float),
            "elapsed": np.where(active, (T - s[j]) / _NS_MIN, 0.0),
            "since_end": np.where(active, 0.0, np.where(has, (T - e[j]) / _NS_MIN, np.nan)),
        }
        for w in (1, 3, 24, 168):
            res[f"starts_{w}h"] = (n_before - np.searchsorted(s, T - w * _NS_H, side="left")).astype(float)
        cols[ob] = res

    keys = next(iter(cols.values())).keys()
    return {k: pd.DataFrame({ob: cols[ob][k] for ob in cols}, index=origins) for k in keys}


# ---------------------------------------------------------------- цілі
def make_target(A: pd.DataFrame, h: int, kind: str) -> pd.DataFrame:
    """window: max(рядки t..t+h−1); point: рядок t+h−1. NaN у хвості, де вікно виходить за сітку."""
    if kind == "window":
        return pd.DataFrame(np.maximum.reduce([A.shift(-k).to_numpy() for k in range(h)]),
                            index=A.index, columns=A.columns)
    if kind == "point":
        return A.shift(-(h - 1))
    raise ValueError(kind)


def climatology(target: pd.DataFrame, h: int, days: int = CLIM_DAYS, m: float = 2.0) -> pd.DataFrame:
    """
    Частота цілі в ту саму годину доби за останні `days` днів, лише з цілей,
    що вже повністю спостережені: ціль origin s відома в s+h ≤ t  ⇔  зсув 24·d ≥ h.
    m-оцінка зі зсувом до загальної частоти за 28 днів (щоб не було 0/1 → log-loss = ∞).
    """
    assert h <= 24, "для h > 24 зсув 24·d з d=1 заглядав би в майбутнє"
    S = sum(target.shift(24 * d) for d in range(1, days + 1))
    prior = target.shift(h).rolling(24 * days, min_periods=24).mean()
    return (S + m * prior) / (days + m)


def online_markov(target: pd.DataFrame, active: pd.DataFrame, h: int,
                  days: int = CLIM_DAYS, m: float = 2.0) -> pd.DataFrame:
    """
    Онлайн-Markov: P(target | active_now) за останні `days` днів для кожної області.
    Беремо лише origin s ≤ t−h (ціль уже спостережена) → shift(h).
    Адаптується до рівня тривог (на відміну від таблиці, вивченої на train) —
    сильний бейзлайн під дрейф і база для offset-моделі.
    """
    w = 24 * days
    yt, at = target.shift(h), active.shift(h)
    roll = lambda x: x.rolling(w, min_periods=24).sum()
    prior = yt.rolling(w, min_periods=24).mean()
    p1 = (roll(yt * at) + m * prior) / (roll(at) + m)
    p0 = (roll(yt * (1 - at)) + m * prior) / (roll(1 - at) + m)
    return active * p1 + (1 - active) * p0


# ---------------------------------------------------------------- датасет
def make_features(hourly: pd.DataFrame, intervals: pd.DataFrame, h: int,
                  kind: str = "window", scope: list[str] = SCOPE,
                  start: pd.Timestamp = REFORM_END) -> pd.DataFrame:
    """
    -> довгий датасет: один рядок = (oblast, origin t), ознаки L1+L1.5, `target`, `clim`.
    hourly — вихід data_loader (усі області: потрібні для сусідів і глобальних ознак).
    """
    hourly = hourly[hourly["hour_utc"] >= start - pd.Timedelta(days=HISTORY_DAYS)]
    A = hourly.pivot(index="hour_utc", columns="oblast", values="alert").astype(float)
    Mn = hourly.pivot(index="hour_utc", columns="oblast", values="alert_minutes").astype(float)
    origins = A.index
    st = interval_state(intervals, origins)
    st = {k: v.reindex(columns=A.columns, fill_value=0.0) for k, v in st.items()}

    past = A.shift(1)                                    # рядок t−1 = [t−1h, t)
    target = make_target(A, h, kind)
    clim = climatology(target, h)
    markov28 = online_markov(target, st["active"], h)

    # глобальні ознаки (всі області)
    n_alert_lasthour = past.sum(axis=1, min_count=1)
    n_active = st["active"].sum(axis=1)
    mass_row = A.sum(axis=1) >= MASS_THRESHOLD           # масована атака в рядку r
    mass_past = mass_row.shift(1, fill_value=False)      # для origin t — рядок t−1
    pos = np.arange(len(origins), dtype=float)
    last_mass = pd.Series(np.where(mass_past, pos, np.nan), index=origins).ffill()
    hours_since_mass = (pos - last_mass).fillna(24 * HISTORY_DAYS)
    mass_onset = (mass_row & ~mass_row.shift(1, fill_value=False)).astype(float)
    mass_count_14d = mass_onset.shift(1).rolling(24 * 14, min_periods=1).sum()

    local_t = origins.tz_convert(KYIV_TZ)
    local_tgt = (origins + (h - 1) * H).tz_convert(KYIV_TZ)   # календар цільової години — детермінований

    frames = []
    for ob in scope:
        nb = NEIGHBORS[ob]
        d = {
            "hour_utc": origins,
            # L1
            "active_now": st["active"][ob],
            "elapsed_min": st["elapsed"][ob],
            "log_since_end_min": np.log1p(st["since_end"][ob].fillna(24 * 60 * HISTORY_DAYS)),
            "starts_24h": st["starts_24h"][ob],
            "starts_7d": st["starts_168h"][ob],
            **{f"lag_{k}": A[ob].shift(k) for k in LAGS},
            **{f"min_lag_{k}": Mn[ob].shift(k) / 60 for k in (1, 2, 3)},
            **{f"roll_{w}": past[ob].rolling(w).mean() for w in (6, 24, 168, 336)},
            "clim": clim[ob],
            "markov28": markov28[ob],
            "hour_t": local_t.hour, "dow_t": local_t.dayofweek,
            "hour_target": local_tgt.hour, "dow_target": local_tgt.dayofweek,
            # L1.5
            "nb_active_n": st["active"][nb].sum(axis=1),
            "nb_starts_1h": st["starts_1h"][nb].sum(axis=1),
            "nb_starts_3h": st["starts_3h"][nb].sum(axis=1),
            "nb_alert_3h": pd.concat([A[nb].shift(k) for k in (1, 2, 3)], axis=1).mean(axis=1),
            "n_active_now": n_active,
            "d_n_active": n_active - n_active.shift(1),
            "n_alert_lasthour": n_alert_lasthour,
            "n_alert_max_6h": n_alert_lasthour.rolling(6).max(),
            "hours_since_mass": hours_since_mass,
            "mass_count_14d": mass_count_14d,
            "target": target[ob],
        }
        f = pd.DataFrame(d, index=origins).reset_index(drop=True)
        f["oblast"] = ob
        frames.append(f)

    df = pd.concat(frames, ignore_index=True)
    df["oblast"] = pd.Categorical(df["oblast"], categories=scope)
    df = df[df["hour_utc"] >= start].dropna(subset=["target", "clim", "markov28"])
    return df.reset_index(drop=True)
