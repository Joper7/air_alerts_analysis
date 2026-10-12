"""
Ракетні удари за ранковими підсумками @kpszsu (data/raw/telegram/kpszsu.parquet).

1) parse_missiles — кількість ракет за ніч (усього / балістичні / крилаті) з речення
   «У ніч на DD місяця … противник атакував <ракети> … <N> ударними БпЛА …».
2) Накопичення («запас ракет»): чи більші масовані удари після довшої паузи; ризик (hazard)
   масованого удару залежно від днів після попереднього.
3) P(ракетний удар цієї ночі): прогноз о 15:00 UTC (≈18:00 Київ) напередодні ночі.
   Відомо на цей момент: підсумки попередніх ночей (публікуються до ~09:00 UTC) і пости @kpszsu
   з available_at < 15:00 UTC. Walk-forward: логістична регресія перенавчається щодня лише на минулих ночах.
   Бейзлайн — онлайн-частота за 28 попередніх ночей.

Запуск: python -m src.missiles  → reports/missiles.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.launches_eda import MON, parse_launches

WORDS = {"однією": 1, "одною": 1, "двома": 2, "трьома": 3, "чотирма": 4, "п'ятьма": 5, "п’ятьма": 5,
         "шістьма": 6, "сімома": 7, "сьома": 7, "вісьмома": 8, "дев'ятьма": 9, "дев’ятьма": 9, "десятьма": 10}
_ATTACK = re.compile(r"противник атакував(.*?)(?:Повітряний напад|$)", re.S)
_NIGHT = re.compile(r"[Уу] ніч на (\d{1,2})\s+(\w+)")
# <кількість> <до 4 слів> ракет…   (кількість — число, «62-ма» або слово; без числа = 1)
_MISSILE = re.compile(r"(?:(\d+)(?:-?[а-яіїє]{1,3})?|(" + "|".join(WORDS) + r"))?\s*((?:[\w\-/«»\"“”]+\s+){0,4}?)ракет\w*",
                      re.I)
MASSIVE = 10                     # ≥10 ракет за ніч = масований ракетний удар
FORECAST_HOUR_UTC = 15


def parse_missiles(posts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in posts[posts["text"].str.contains("атакува", case=False)].iterrows():
        t = r["text"].replace("\xa0", " ").replace("\n", " ")
        mn, ma = _NIGHT.search(t), _ATTACK.search(t)
        if not (mn and ma and mn.group(2) in MON):
            continue
        seg = ma.group(1)
        seg = seg[: seg.find("ударн")] if "ударн" in seg else seg          # лише частина ДО дронів
        tot = bal = cru = 0
        for m in _MISSILE.finditer(seg):
            n = 1                                                     # без кількості («балістичною ракетою») = 1
            for tok in m.group(0).split():                            # перший токен-число або числівник
                tok = tok.lower()
                if re.fullmatch(r"\d+(?:-?[а-яіїє]{1,3})?", tok):
                    n = int(re.match(r"\d+", tok).group(0)); break
                if tok in WORDS:
                    n = WORDS[tok]; break
            ctx = (m.group(3) or "") + seg[m.end(): m.end() + 40]
            if re.search(r"зенітн", ctx, re.I):                            # ЗКР С-300/400 — рахуємо як балістику
                bal += n
            elif re.search(r"балістичн|іскандер|кинджал|кн-23", ctx, re.I):
                bal += n
            elif re.search(r"крилат|х-101|х-555|калібр|х-22|х-32|х-59|х-69|оникс|циркон", ctx, re.I):
                cru += n
            tot += n
        mon = MON[mn.group(2)]
        year = r["date_utc"].year - (1 if mon == 12 and r["date_utc"].month == 1 else 0)
        rows.append({"night": pd.Timestamp(year, mon, int(mn.group(1))), "missiles": tot,
                     "ballistic": bal, "cruise": cru, "available_at": r["available_at"]})
    df = pd.DataFrame(rows).sort_values("available_at").drop_duplicates("night", keep="first")
    return df.set_index("night").sort_index()


def nightly_table(posts: pd.DataFrame) -> pd.DataFrame:
    """Одна строка = ніч «на D». Ночі без підсумку → 0 ракет (підсумок не виходить лише в тихі ночі)."""
    m = parse_missiles(posts)
    idx = pd.date_range(m.index.min(), m.index.max(), freq="D")
    t = m[["missiles", "ballistic", "cruise"]].reindex(idx).fillna(0)
    t["drones"] = parse_launches(posts).reindex(idx).fillna(0)
    return t


def accumulation(t: pd.DataFrame) -> list[str]:
    mas = t[t["missiles"] >= MASSIVE].copy()
    mas["gap_days"] = mas.index.to_series().diff().dt.days
    x = mas.dropna()
    rho = x[["gap_days", "missiles"]].corr(method="spearman").iloc[0, 1]
    # ризик масованого удару залежно від днів після попереднього
    since = []
    last = None
    for d, v in t["missiles"].items():
        since.append(np.nan if last is None else (d - last).days)
        if v >= MASSIVE:
            last = d
    t2 = t.assign(since=since, massive=(t["missiles"] >= MASSIVE).astype(int)).dropna()
    bins = pd.cut(t2["since"], [0, 3, 7, 10, 14, 21, 60], right=True)
    hz = t2.groupby(bins, observed=True)["massive"].agg(["mean", "size"]).round(3)
    return ["## Накопичення («запас ракет»)", "",
            f"Масованих ударів (≥{MASSIVE} ракет за ніч): {len(mas)}; медіанна пауза між ними — "
            f"{x['gap_days'].median():.0f} діб, медіанний розмір — {x['missiles'].median():.0f} ракет.", "",
            f"Spearman(пауза перед ударом, розмір удару) = **{rho:+.2f}** (n={len(x)}). "
            "Накопичення дало б ρ > 0: що довша пауза, то більший удар.", "",
            "Ризик масованого удару в конкретну ніч залежно від діб після попереднього:", "",
            hz.rename(columns={"mean": "P(масований)", "size": "ночей"}).to_markdown(), ""]


def alert_duration_by_type(t: pd.DataFrame, intervals: pd.DataFrame, scope: list[str]) -> list[str]:
    """
    Наївна довідка (НЕ для навчання: тип ночі відомий лише з ранкового підсумку, тобто після події):
    медіанна тривалість і к-сть тривог у скоупі в ніч 18:00–09:00 Київ залежно від типу атаки.
    """
    tz = "Europe/Kyiv"
    iv = intervals[intervals["oblast"].isin(scope)].copy()
    loc = iv["started_at"].dt.tz_convert(tz)
    # ніч «на D» = старт тривоги з 18:00 D−1 до 09:00 D (за Києвом)
    iv["night"] = (loc + pd.Timedelta(hours=6)).dt.normalize().dt.tz_localize(None)
    iv = iv[(loc.dt.hour >= 18) | (loc.dt.hour < 9)]
    iv["dur_min"] = (iv["finished_at"] - iv["started_at"]).dt.total_seconds() / 60
    iv = iv[~iv["naive"]] if "naive" in iv else iv
    typ = np.select([t["ballistic"] > 0, t["cruise"] > 0, t["missiles"] >= MASSIVE, t["drones"] > 0],
                    ["балістика", "крилаті", "масований", "лише дрони"], "без підсумку")
    tt = pd.Series(typ, index=t.index, name="type")
    j = iv.join(tt, on="night").dropna(subset=["type"])
    g = j.groupby(["type", "oblast"])["dur_min"].median().unstack().round(0)
    g["усі 4 (медіана)"] = j.groupby("type")["dur_min"].median().round(0)
    g["тривог"] = j.groupby("type").size()
    return ["## Тривалість нічних тривог за типом атаки (наївна довідка)", "",
            "Тип ночі — з ранкового підсумку @kpszsu (відомий лише ПІСЛЯ ночі → для навчання не придатний). "
            "Тривоги — volunteer, старт 18:00–09:00 за Києвом, без `naive`. Медіана тривалості, хв:", "",
            g.to_markdown(), ""]


def daily_features(t: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    """Ознаки для ночі «на D», відомі о 15:00 UTC дня D−1 (лише підсумки ночей ≤ D−1)."""
    f = pd.DataFrame(index=t.index)
    m, past = t["missiles"], t.shift(1)                       # past — ніч D−1 і раніше
    any_ = (past["missiles"] > 0).astype(float)
    mas = (past["missiles"] >= MASSIVE).astype(float)
    pos = np.arange(len(t), dtype=float)
    for name, s in (("any", any_), ("massive", mas)):
        last = pd.Series(np.where(s > 0, pos, np.nan), index=t.index).ffill()
        f[f"days_since_{name}"] = (pos - last).fillna(60).clip(upper=60)
    f["any_7"], f["any_30"] = any_.rolling(7, min_periods=1).sum(), any_.rolling(30, min_periods=1).sum()
    f["massive_30"] = mas.rolling(30, min_periods=1).sum()
    f["missiles_30"] = past["missiles"].rolling(30, min_periods=1).sum()
    f["drones_last"] = past["drones"]
    f["drones_rel7"] = past["drones"].rolling(7, min_periods=1).mean() / (past["drones"].rolling(60, min_periods=7).mean() + 1)
    f["dow"] = t.index.dayofweek
    # пости до 15:00 UTC дня D−1: стратегічна авіація / ракетні загрози за 24 год і 72 год
    p = posts.assign(av=pd.to_datetime(posts["available_at"], utc=True))
    low = p["text"].str.lower()
    av = p.loc[low.str.contains(r"ту-?95|ту-?160|ту-?22|міг-?31|стратегічн|носі[їя]|кинджал|калібр|пуск", regex=True), "av"]
    bal = p.loc[low.str.contains(r"балістик|балістичн|швидкісн", regex=True), "av"]
    T = (t.index.tz_localize("UTC") - pd.Timedelta(days=1) + pd.Timedelta(hours=FORECAST_HOUR_UTC)).as_unit("ns").asi8
    for nm, s in (("aviation", av), ("ballistic_posts", bal)):
        ts = np.sort(pd.DatetimeIndex(s).as_unit("ns").asi8)
        for w in (24, 72):
            f[f"{nm}_{w}h"] = np.searchsorted(ts, T) - np.searchsorted(ts, T - w * 3_600_000_000_000)
    return f.fillna(0)                                           # перша ніч ряду не має «попередньої»


def walk_forward(f: pd.DataFrame, y: pd.Series, start: str = "2026-02-01") -> pd.DataFrame:
    rows = []
    for d in f.index[f.index >= start]:
        tr = f.index < d                                         # лише ночі, що вже минули
        if y[tr].nunique() < 2:
            continue
        base = y[tr].iloc[-28:]
        p0 = (base.sum() + 1) / (len(base) + 2)                  # онлайн-частота (Лаплас)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        clf.fit(f[tr].iloc[-180:], y[tr].iloc[-180:])
        rows.append({"night": d, "y": y[d], "base": p0, "model": clf.predict_proba(f.loc[[d]])[0, 1]})
    return pd.DataFrame(rows).set_index("night")


def score(r: pd.DataFrame, seed: int = 0) -> str:
    b = lambda c: ((r[c] - r["y"]) ** 2)
    d = (b("model") - b("base")).to_numpy()
    wk = pd.Series(d, index=r.index).groupby(r.index.to_period("W")).sum().to_numpy()
    cnt = pd.Series(1, index=r.index).groupby(r.index.to_period("W")).sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), (3000, len(wk)))
    boots = wk[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    auc = roc_auc_score(r["y"], r["model"]) if r["y"].nunique() == 2 else float("nan")
    auc0 = roc_auc_score(r["y"], r["base"]) if r["y"].nunique() == 2 else float("nan")
    return (f"ночей {len(r)}, частка {r['y'].mean():.3f} | Brier модель {b('model').mean():.4f}, онлайн-частота "
            f"{b('base').mean():.4f} | BSS {1 - b('model').mean() / b('base').mean():+.3f} "
            f"(ΔBrier 95% CI [{lo:+.4f}; {hi:+.4f}]) | AUC {auc:.3f} (частота {auc0:.3f})")


def main() -> None:
    posts = pd.read_parquet("data/raw/telegram/kpszsu.parquet")
    t = nightly_table(posts)
    t = t[t.index >= "2026-01-01"]                               # з 01.2026 формат підсумків стабільний
    lines = ["# Ракетні удари (@kpszsu, ранкові підсумки)", "",
             f"Ночей: {len(t)} ({t.index.min():%d.%m.%Y} – {t.index.max():%d.%m.%Y}); з ракетами — "
             f"{(t['missiles'] > 0).mean():.1%}; масованих (≥{MASSIVE}) — {(t['missiles'] >= MASSIVE).sum()}; "
             f"ракет усього {int(t['missiles'].sum())} (балістичних {int(t['ballistic'].sum())}, крилатих {int(t['cruise'].sum())}).", ""]
    lines += accumulation(t)
    iv = pd.read_parquet("data/processed/intervals_volunteer.parquet")
    lines += alert_duration_by_type(t, iv, ["Kyiv City", "Kyivska oblast", "Cherkaska oblast", "Kirovohradska oblast"])
    f = daily_features(t, posts)
    lines += ["## P(ракетний удар цієї ночі) — walk-forward з 01.02.2026", ""]
    for nm, y in (("будь-яка ракета (≥1)", (t["missiles"] > 0).astype(int)),
                  (f"масований удар (≥{MASSIVE})", (t["missiles"] >= MASSIVE).astype(int))):
        r = walk_forward(f, y)
        lines.append(f"- **{nm}:** {score(r)}")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(f, (t["missiles"] > 0).astype(int))
    coef = pd.Series(clf[-1].coef_[0], index=f.columns).sort_values(key=abs, ascending=False)
    lines += ["", "Коефіцієнти логістичної регресії (на всіх ночах, стандартизовані; для інтерпретації):", "",
              coef.round(3).to_frame("coef").to_markdown()]
    Path("reports/missiles.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
