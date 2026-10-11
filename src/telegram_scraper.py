"""
Збір повідомлень публічних Telegram-каналів для шару L2 (опційно, після MVP).

Запуск ТІЛЬКИ локально: Telethon логіниться твоїм номером телефону.
  1. API-ключі: https://my.telegram.org/apps
  2. Windows PowerShell:
       $env:TG_API_ID="12345"; $env:TG_API_HASH="abc..."
     Linux/macOS:
       export TG_API_ID=12345 TG_API_HASH=abc...
  3. pip install telethon pandas pyarrow
  4. python -m src.telegram_scraper kpszsu --since 2025-11-01
     (перший запуск попросить телефон і код з Telegram)

!!! air_alerts_session.session = повний доступ до акаунта. Він у .gitignore — НЕ комітити.

Вихід: data/raw/telegram/<channel>.parquet
  channel, msg_id, date_utc, edit_date_utc, available_at, is_forward, text

Правила проти leakage (застосовувати у text_features.py):
  - фіча в момент t бере лише пости з available_at < t − 5 хв;
  - available_at = edit_date, якщо пост редагували (першої версії тексту API не віддає);
  - views НЕ збираємо: це лічильник на момент скрапу, а не на момент t;
  - видалені пости відсутні (survivorship bias) — описати в обмеженнях;
  - для forward беремо дату репосту в цьому каналі (m.date), не оригіналу.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd
from telethon import TelegramClient

OUT_DIR = Path("data/raw/telegram")
COLUMNS = ["channel", "msg_id", "date_utc", "edit_date_utc", "available_at", "is_forward", "text"]


async def fetch_channel(client: TelegramClient, channel: str, since: pd.Timestamp) -> pd.DataFrame:
    rows = []
    # iter_messages іде від найновіших до старіших; зупиняємось на `since`.
    # wait_time — пауза між батчами, щоб не ловити FloodWait на довгій історії.
    async for m in client.iter_messages(channel, wait_time=1):
        if m.date < since:
            break
        if not m.message:            # медіа без тексту/підпису
            continue
        rows.append({
            "channel": channel,
            "msg_id": m.id,
            "date_utc": m.date,                     # tz-aware UTC
            "edit_date_utc": m.edit_date,
            "is_forward": m.fwd_from is not None,
            "text": m.message,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in ("date_utc", "edit_date_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["available_at"] = df["edit_date_utc"].fillna(df["date_utc"])  # консервативна доступність
    return df.sort_values("date_utc")[COLUMNS].reset_index(drop=True)


async def main(channels: list[str], since: str) -> None:
    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    since_ts = pd.Timestamp(since, tz="UTC")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # flood_sleep_threshold: автоматично чекати FloodWait до 5 хв замість падіння
    async with TelegramClient("air_alerts_session", api_id, api_hash,
                              flood_sleep_threshold=300) as client:
        for ch in channels:
            ch = ch.lstrip("@")
            try:
                df = await fetch_channel(client, ch, since_ts)
            except Exception as e:   # неіснуючий/приватний хендл не валить решту каналів
                print(f"{ch}: ПРОПУЩЕНО ({type(e).__name__}: {e})")
                continue
            path = OUT_DIR / f"{ch}.parquet"
            df.to_parquet(path, index=False)
            span = (f"{df['date_utc'].min():%Y-%m-%d} — {df['date_utc'].max():%Y-%m-%d}"
                    if len(df) else "порожньо")
            print(f"{ch}: {len(df)} повідомлень ({span}) -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("channels", nargs="+", help="username каналів (з @ або без)")
    p.add_argument("--since", default="2025-11-01", help="UTC-дата, з якої збирати")
    a = p.parse_args()
    asyncio.run(main(a.channels, a.since))
