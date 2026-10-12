"""
Збір історії публічного каналу через веб-прев'ю t.me/s/<channel> — БЕЗ логіну й API-ключів
(альтернатива src/telegram_scraper.py, який потребує Telethon і телефон).

Сторінка t.me/s/<channel>?before=<id> віддає ~20 постів з id < before:
  id, час публікації (UTC), текст, позначка «edited».

Правила проти leakage (застосовуються в src/text_features.py):
  - available_at = час публікації; пост з позначкою «edited» має невідомий час
    редагування → поточний текст міг з'явитися пізніше → такі пости ВІДКИДАЄМО;
  - фіча в момент t бере лише пости з available_at < t − 5 хв;
  - видалені пости відсутні (survivorship bias) — описати в обмеженнях.

Запуск:  python -m src.telegram_web kpszsu --since 2025-10-25
Вихід:   data/raw/telegram/<channel>_web.parquet (дописується інкрементно, можна перезапускати)
"""
from __future__ import annotations

import argparse
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("data/raw/telegram")
UA = {"User-Agent": "Mozilla/5.0 (research; air-alerts pet project)"}
_BLOCK = re.compile(r'<div class="tgme_widget_message_wrap.*?(?=<div class="tgme_widget_message_wrap|\Z)', re.S)
_ID = re.compile(r'data-post="[^/"]+/(\d+)"')
_TIME = re.compile(r'<time[^>]*datetime="([^"]+)"')
_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_EDITED = re.compile(r'>\s*edited\s*<')


def _clean(t: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", t)
    return html.unescape(re.sub(r"<[^>]+>", "", t)).strip()


def parse_page(s: str) -> list[dict]:
    rows = []
    for b in _BLOCK.findall(s):
        mid, tm = _ID.search(b), _TIME.search(b)
        if not (mid and tm):
            continue
        tx = _TEXT.search(b)
        rows.append({"msg_id": int(mid.group(1)), "date_utc": tm.group(1),
                     "edited": bool(_EDITED.search(b)), "text": _clean(tx.group(1)) if tx else ""})
    return rows


def fetch(channel: str, before: int | None, session: requests.Session, tries: int = 6) -> str:
    url = f"https://t.me/s/{channel}" + (f"?before={before}" if before else "")
    for k in range(tries):
        try:
            r = session.get(url, headers=UA, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(30 * (k + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"не вдалося отримати {url}")


def scrape(channel: str, since: str, pause: float = 1.0) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{channel}_web.parquet"
    since_ts = pd.Timestamp(since, tz="UTC")
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    before = int(old["msg_id"].min()) if len(old) else None    # продовжуємо вниз від найстарішого
    sess, buf, n = requests.Session(), [], 0
    while True:
        rows = parse_page(fetch(channel, before, sess))
        if not rows:
            break
        buf += rows
        oldest = min(rows, key=lambda r: r["msg_id"])
        before = oldest["msg_id"]
        n += 1
        if n % 25 == 0:
            old = _save(old, buf, path); buf = []
            print(f"{n} стор., найстаріший {oldest['date_utc']} (id {before}), всього {len(old)}", flush=True)
        if pd.Timestamp(oldest["date_utc"]) < since_ts:
            break
        time.sleep(pause)
    old = _save(old, buf, path)
    print(f"готово: {len(old)} постів, {old['date_utc'].min()} — {old['date_utc'].max()}")
    return path


def _save(old: pd.DataFrame, rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    if df.empty:
        return df
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    df = df.drop_duplicates("msg_id").sort_values("msg_id").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("channel")
    p.add_argument("--since", default="2025-10-25")
    p.add_argument("--pause", type=float, default=1.0)
    a = p.parse_args()
    scrape(a.channel.lstrip("@"), a.since, a.pause)
