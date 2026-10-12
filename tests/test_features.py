"""
Тести вирівнювання часу та відсутності leakage.

  pytest -q tests/
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import to_hourly
from src.evaluate import chrono_split
from src.features import FEATURE_SETS, NEIGHBORS, make_features, make_target
from src.text_features import TG_COLS, tg_features

PROC = Path("data/processed")
U = lambda s: pd.Timestamp(s, tz="UTC")


# ---------------------------------------------------------------- синтетика
def test_target_alignment():
    """Тривога 10:20–11:10 → рядки 10 і 11 = 1; window h=1 в origin 10:00 = 1, в 09:00 = 0."""
    iv = pd.DataFrame({"oblast": ["X"], "started_at": [U("2026-01-01 10:20")],
                       "finished_at": [U("2026-01-01 11:10")]})
    hourly = to_hourly(iv, end=U("2026-01-01 14:00"))
    A = hourly.pivot(index="hour_utc", columns="oblast", values="alert").astype(float)
    assert A.loc[U("2026-01-01 10:00"), "X"] == 1 and A.loc[U("2026-01-01 11:00"), "X"] == 1
    w1 = make_target(A, 1, "window")["X"]
    w3 = make_target(A, 3, "window")["X"]
    p3 = make_target(A, 3, "point")["X"]
    assert w1[U("2026-01-01 10:00")] == 1 and w1[U("2026-01-01 12:00")] == 0
    # origin 10:00, h=3: вікно 10..12 → 1; point = рядок 12 → 0
    assert w3[U("2026-01-01 10:00")] == 1 and p3[U("2026-01-01 10:00")] == 0
    # останні h−1 origin без повного вікна → NaN
    assert np.isnan(w3.iloc[-1]) and np.isnan(w3.iloc[-2])


def test_split_purge():
    t = pd.date_range("2026-01-01", "2026-04-01", freq="h", tz="UTC", inclusive="left")
    df = pd.DataFrame({"hour_utc": t})
    for h in (1, 3, 24):
        masks, b = chrono_split(df, h)
        assert (df.loc[masks["train"], "hour_utc"] + h * pd.Timedelta(hours=1) <= b["val_start"]).all()
        assert (df.loc[masks["val"], "hour_utc"] + h * pd.Timedelta(hours=1) <= b["test_start"]).all()
        assert not (masks["train"] & masks["val"]).any() and not (masks["val"] & masks["test"]).any()


# ---------------------------------------------------------------- реальні дані: тест обрізання
@pytest.mark.skipif(not (PROC / "intervals_volunteer.parquet").exists(),
                    reason="спершу python -m src.data_loader")
@pytest.mark.parametrize("h", [1, 3])
def test_no_leakage_truncation(h):
    """
    Будуємо ознаки двічі: (а) з усіх даних; (б) з даних, які були б відомі в момент t0
    (тривоги зі стартом < t0; ті, що ще тривають, обрізані до t0+1с — «досі триває»).
    Для всіх origin ≤ t0 кожна ознака має збігтися. Інакше — leakage.
    """
    iv = pd.read_parquet(PROC / "intervals_volunteer.parquet")
    start = U("2026-05-01")
    iv = iv[iv["finished_at"] >= start - pd.Timedelta(days=70)]
    end = U("2026-07-01")
    t0 = U("2026-06-15 21:00")            # вечір: багато активних тривог

    full_h = to_hourly(iv, end=end)
    full = make_features(full_h, iv, h, start=start)

    cut = iv[iv["started_at"] < t0].copy()
    cut["finished_at"] = cut["finished_at"].clip(upper=t0 + pd.Timedelta(seconds=1))
    cut_h = to_hourly(cut, end=end)
    trunc = make_features(cut_h, cut, h, start=start)

    cols = sorted({c for fs in FEATURE_SETS.values() for c in fs} - {"oblast"} - set(TG_COLS))
    key = ["oblast", "hour_utc"]
    a = full[full["hour_utc"] <= t0].set_index(key)[cols].sort_index()
    b = trunc[trunc["hour_utc"] <= t0].set_index(key)[cols].sort_index()
    assert len(a) > 1000 and a.index.equals(b.index)
    diff = (a - b).abs().max()
    assert (diff < 1e-9).all(), f"leakage у ознаках: {diff[diff >= 1e-9].to_dict()}"
    # санітарна перевірка: після t0 ознаки МАЮТЬ розходитись, інакше обрізання не спрацювало
    later = lambda d: d[d["hour_utc"] > t0 + pd.Timedelta(hours=24)].set_index(key)[cols].sort_index()
    assert ((later(full) - later(trunc)).abs().max() > 0).any()


# ---------------------------------------------------------------- Telegram: буфер і обрізання
def test_tg_buffer_and_truncation():
    """Пост за 3 хв до t НЕ рахується (буфер 5 хв); за 10 хв — рахується; пости після t0 не впливають."""
    t = U("2026-01-01 12:00")
    posts = pd.DataFrame({
        "msg_id": [1, 2, 3, 4],
        "date_utc": [t - pd.Timedelta(minutes=10), t - pd.Timedelta(minutes=3),
                     t + pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=20)],
        "edited": [False, False, False, True],
        "text": ["🛵 БпЛА курсом на Бровари", "🛵 БпЛА на Київ", "🚀 Балістика на Київ", "🛵 Біла Церква"],
    })
    origins = pd.DatetimeIndex([t, t + pd.Timedelta(hours=1)])
    g, per = tg_features(posts, origins, ["Kyiv City", "Kyivska oblast"], NEIGHBORS)
    assert g.loc[t, "tg_n_30m"] == 1                       # лише пост №1 (№2 у буфері, №4 edited)
    assert per["Kyivska oblast"].loc[t, "tg_self_30m"] == 1 and per["Kyiv City"].loc[t, "tg_self_30m"] == 0
    assert g.loc[t, "tg_ballistic_1h"] == 0                # пост з майбутнього не видно
    g2, per2 = tg_features(posts[posts["date_utc"] < t], origins[:1], ["Kyiv City", "Kyivska oblast"], NEIGHBORS)
    assert (g2.loc[t] == g.loc[t]).all() and (per2["Kyiv City"].loc[t] == per["Kyiv City"].loc[t]).all()
