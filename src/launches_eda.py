"""
Гіпотеза «запасу дронів»: якщо вчора/цього тижня запустили багато, сьогодні/наступного тижня
має бути менше (запас вичерпано), з поправкою на тренд виробництва.

Дані: ранкові підсумки @kpszsu «У ніч на DD місяця … противник атакував N ударними БпЛА …»
(data/raw/telegram/kpszsu.parquet). Підсумок публікується вранці → для прогнозу наступних ночей легальний.
«Виробництво» = середній темп запусків за попередні 60 ночей (лише минуле).

Запуск: python -m src.launches_eda  → reports/launches_eda.md
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MON = {"січня": 1, "лютого": 2, "березня": 3, "квітня": 4, "травня": 5, "червня": 6, "липня": 7,
       "серпня": 8, "вересня": 9, "жовтня": 10, "листопада": 11, "грудня": 12}
_NIGHT = re.compile(r"[Уу] ніч на (\d{1,2})\s+(\w+)")
_N = re.compile(r"(\d+)\s+ударн\w*\s+(?:БпЛА|безпілот)")


def parse_launches(posts: pd.DataFrame) -> pd.Series:
    """Кількість ударних БпЛА за ніч (індекс — дата ранку після ночі)."""
    rows = []
    for _, r in posts[posts["text"].str.contains("атакува", case=False)].iterrows():
        t = r["text"].replace("\xa0", " ")
        m, n = _NIGHT.search(t), _N.search(t)
        if not (m and n and m.group(2) in MON):
            continue
        mon = MON[m.group(2)]
        year = r["date_utc"].year - (1 if mon == 12 and r["date_utc"].month == 1 else 0)
        rows.append({"night": pd.Timestamp(year, mon, int(m.group(1))), "launched": int(n.group(1)),
                     "available_at": r["available_at"]})
    df = pd.DataFrame(rows).sort_values("available_at").drop_duplicates("night", keep="first")
    s = df.set_index("night")["launched"].sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).fillna(0)


def main() -> None:
    s = parse_launches(pd.read_parquet("data/raw/telegram/kpszsu.parquet"))
    s = s[s.index >= "2026-01-01"]          # з 01.2026 формат підсумків стабільний
    trend = s.shift(1).rolling(60, min_periods=30).mean()
    lines = ["# Гіпотеза «запасу дронів»", "",
             "Spearman ρ між перевитратою за k попередніх ночей (відносно темпу за 60 ночей) і "
             "інтенсивністю наступних m ночей. **Виснаження запасу дало б ρ < 0.**", "",
             "| k ночей до t | m=1 | m=7 |", "|---|---|---|"]
    for k in (1, 3, 7, 14, 21):
        exc = (s.shift(1).rolling(k).sum() - k * trend) / (k * trend)
        rho = []
        for m in (1, 7):
            fut = s[::-1].rolling(m).mean()[::-1] / trend
            rho.append(pd.concat([exc, fut], axis=1).dropna().corr(method="spearman").iloc[0, 1])
        lines.append(f"| {k} | {rho[0]:+.3f} | {rho[1]:+.3f} |")
    stock = (trend - s).shift(1).rolling(30).sum() / trend
    fut7 = s[::-1].rolling(7).mean()[::-1] / trend
    X = pd.concat([stock.rename("stock"), fut7.rename("fut")], axis=1).dropna()
    X["q"] = pd.qcut(X["stock"], 4, labels=["витратили найбільше", "Q2", "Q3", "накопичили найбільше"])
    lines += ["", "Модель запасу: Σ(темп − запуски) за 30 ночей → інтенсивність наступного тижня / темп:", "",
              X.groupby("q", observed=True)["fut"].agg(["mean", "size"]).round(2).to_markdown(), "",
              "Темп запусків (середнє за 60 ночей) на кінець місяця: "
              + ", ".join(f"{k:%m.%Y} — {v:.0f}" for k, v in trend.resample("MS").last().dropna().items())]
    Path("reports/launches_eda.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
