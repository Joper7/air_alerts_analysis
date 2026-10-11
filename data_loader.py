"""
Крок 1: завантаження та очищення даних про повітряні тривоги.

Джерело: github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT, оновлюється щодня)
  - official_data_en.csv  : офіційні тривоги, рівні oblast / raion / hromada, з 2022-03-15
  - volunteer_data_en.csv : дані eTryvoga, ТІЛЬКИ рівень області, з 2022-02-25
Усі мітки часу в джерелі — UTC.

Ціль однакова для всього періоду: "у будь-якій частині області діє тривога" =
union інтервалів усіх рівнів усередині області. Реформа oblast -> raion
(07–11.2025) все одно дає структурний злам -> тренуємо з REFORM_END.

Джерела НЕ склеюємо (згода лише ~82% на перетині 06–09.2026).
За замовчуванням — volunteer: він найсвіжіший.

Використання:
    python -m src.data_loader                 # volunteer, кешований CSV
    python -m src.data_loader volunteer --force   # перекачати свіжі CSV (фінальний прогін!)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = ("https://raw.githubusercontent.com/Vadimkin/"
            "ukrainian-air-raid-sirens-dataset/main/datasets/")
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
KYIV_TZ = "Europe/Kyiv"

SOURCE_FILES = {"volunteer": "volunteer_data_en.csv", "official": "official_data_en.csv"}

# Постійні тривоги, яких немає в датасеті (див. README джерела) — виключаємо
PERMANENT_ALERT_REGIONS = {"Luhanska oblast", "Avtonomna Respublika Krym"}

MAX_DURATION_H = 24  # довші інтервали вважаємо артефактами (пропущений відбій)

# Реформа йшла поступово (Київщина з 07.2025, захід — до 06.11.2025).
# Дата, після якої районний рівень домінує у всіх областях.
REFORM_END = pd.Timestamp("2025-11-07", tz="UTC")


# ---------------------------------------------------------------- download
def download(name: str, force: bool = False) -> Path:
    """Завантажує CSV у data/raw (кешує локально)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if force or not path.exists():
        df = pd.read_csv(BASE_URL + name)
        df.to_csv(path, index=False)
    return path


def snapshot_time(name: str) -> pd.Timestamp:
    """Момент, коли CSV було скачано (mtime файлу) = 'зараз' для цього знімка даних."""
    path = RAW_DIR / name
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").floor("h")


# ---------------------------------------------------------------- cleaning
def _parse_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    return df


def _parse_bool(s: pd.Series) -> pd.Series:
    """Безпечний парсинг True/False: astype(bool) на рядку 'False' дає True."""
    return s.astype(str).str.strip().str.lower().isin({"true", "1"})


def _drop_bad_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Видаляє нульові/від'ємні та аномально довгі (>24 год) інтервали."""
    dur_h = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 3600
    mask = (dur_h > 0) & (dur_h <= MAX_DURATION_H)
    return df.loc[mask].copy()


def merge_intervals(df: pd.DataFrame, key: str = "oblast") -> pd.DataFrame:
    """
    Об'єднує інтервали, що перекриваються, в межах однієї області.
    5 районних тривог, що накладаються, стають одним інтервалом "область під тривогою".

    Якщо є колонка `naive` (volunteer), блок позначається naive=True,
    якщо хоч один його інтервал був без відбою (тривалість штучна) —
    потрібно для Задачі B (такі тривоги = цензуровані).
    """
    has_naive = "naive" in df.columns
    out = []
    for region, g in df.sort_values("started_at").groupby(key, sort=False):
        starts = g["started_at"].to_numpy()
        ends = g["finished_at"].to_numpy()
        # Новий блок починається, коли старт > max(кінців до нього)
        run_max_end = np.maximum.accumulate(ends)
        block_id = np.cumsum(np.r_[True, starts[1:] > run_max_end[:-1]])

        data = {"s": starts, "e": ends, "b": block_id}
        agg = {"started_at": ("s", "min"), "finished_at": ("e", "max")}
        if has_naive:
            data["n"] = g["naive"].to_numpy()
            agg["naive"] = ("n", "max")
        merged = pd.DataFrame(data).groupby("b").agg(**agg)
        merged[key] = region
        out.append(merged)

    res = pd.concat(out, ignore_index=True)
    for col in ("started_at", "finished_at"):           # numpy міг зняти tz
        if res[col].dt.tz is None:
            res[col] = res[col].dt.tz_localize("UTC")
    cols = [key, "started_at", "finished_at"] + (["naive"] if has_naive else [])
    return res[cols].sort_values([key, "started_at"]).reset_index(drop=True)


def load_official(force: bool = False) -> pd.DataFrame:
    df = pd.read_csv(download(SOURCE_FILES["official"], force))
    df = _parse_times(df)
    df = df[~df["oblast"].isin(PERMANENT_ALERT_REGIONS)]
    df = _drop_bad_intervals(df.drop_duplicates())
    return merge_intervals(df, key="oblast")


def load_volunteer(force: bool = False) -> pd.DataFrame:
    """Волонтерські дані: тільки обласний рівень, свіжі (оновлення щодня)."""
    df = pd.read_csv(download(SOURCE_FILES["volunteer"], force))
    df = _parse_times(df).rename(columns={"region": "oblast"})
    df = df[~df["oblast"].isin(PERMANENT_ALERT_REGIONS)]
    df["naive"] = _parse_bool(df["naive"])   # відбою не було, тривалість +30 хв штучно
    df = _drop_bad_intervals(df.drop_duplicates())
    return merge_intervals(df, key="oblast")


# ---------------------------------------------------------------- hourly grid
def to_hourly(intervals: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Інтервали -> погодинна сітка (довгий формат):
        oblast | hour_utc | alert_minutes (0..60) | alert (0/1) | hour_kyiv
    Рядок hour_utc = t описує інтервал [t, t+1h).
    Сітка в UTC — інакше DST дає дубль/пропуск години і ламає shift().
    Сітка спільна для всіх областей, години без тривоги = явні нулі.
    """
    start = intervals["started_at"].min().floor("h")
    end = (end if end is not None else intervals["finished_at"].max()).ceil("h")
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")

    frames = []
    for region, g in intervals.groupby("oblast"):
        minutes = np.zeros(len(hours))
        for s, e in zip(g["started_at"], g["finished_at"]):
            i0 = hours.searchsorted(s.floor("h"))
            i1 = hours.searchsorted(e.ceil("h"))
            for i in range(i0, min(i1, len(hours))):
                h0 = hours[i]
                overlap = (min(e, h0 + pd.Timedelta(hours=1)) - max(s, h0)).total_seconds() / 60
                minutes[i] += max(overlap, 0)
        frames.append(pd.DataFrame({
            "oblast": region, "hour_utc": hours,
            "alert_minutes": np.clip(minutes, 0, 60),
        }))
    hourly = pd.concat(frames, ignore_index=True)
    hourly["alert"] = (hourly["alert_minutes"] > 0).astype("int8")
    hourly["hour_kyiv"] = hourly["hour_utc"].dt.tz_convert(KYIV_TZ)
    return hourly


# ---------------------------------------------------------------- pipeline
def build(source: str = "volunteer", force: bool = False) -> pd.DataFrame:
    """
    -> data/processed/hourly_<source>.parquet, intervals_<source>.parquet, meta_<source>.json

    data_cutoff = момент скачування − 24 год. Тривога, що ще триває в момент
    скачування, у CSV відсутня -> її години виглядали б як 0 (хибні негативи
    в кінці тесту). Обрізаємо хвіст на MAX_DURATION_H.
    """
    loader = {"volunteer": load_volunteer, "official": load_official}[source]
    intervals = loader(force)
    snap = snapshot_time(SOURCE_FILES[source])
    data_cutoff = snap - pd.Timedelta(hours=MAX_DURATION_H)

    hourly = to_hourly(intervals, end=snap)          # тихі години після останньої тривоги = 0
    hourly = hourly[hourly["hour_utc"] < data_cutoff].reset_index(drop=True)
    # post_reform — константа при тренуванні з REFORM_END; лишаємо лише для EDA
    hourly["post_reform"] = (hourly["hour_utc"] >= REFORM_END).astype("int8")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROC_DIR / f"hourly_{source}.parquet", index=False)
    intervals.to_parquet(PROC_DIR / f"intervals_{source}.parquet", index=False)
    meta = {
        "source": source,
        "snapshot_utc": str(snap),
        "data_cutoff_utc": str(data_cutoff),
        "last_hour_utc": str(hourly["hour_utc"].max()),
        "n_intervals": int(len(intervals)),
        "n_naive_blocks": int(intervals["naive"].sum()) if "naive" in intervals else None,
        "oblasts": sorted(hourly["oblast"].unique().tolist()),
    }
    (PROC_DIR / f"meta_{source}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"{source}: {len(intervals)} інтервалів, знімок {snap:%Y-%m-%d %H:%M} UTC, "
          f"сітка до {data_cutoff:%Y-%m-%d %H:%M} UTC (не включно)")
    return hourly


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default="volunteer", choices=list(SOURCE_FILES))
    p.add_argument("--force", action="store_true", help="перекачати свіжі CSV")
    a = p.parse_args()
    h = build(source=a.source, force=a.force)
    print(h.shape)
    print("Області:", sorted(h["oblast"].unique()))
    print(h.groupby("oblast")["alert"].mean().sort_values(ascending=False).round(3).head(10))
