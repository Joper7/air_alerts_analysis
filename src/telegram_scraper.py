"""
Збір повідомлень публічних Telegram-каналів через Telethon (офіційний API, потрібен логін).

Перевага над src/telegram_web.py: API віддає ЧАС РЕДАГУВАННЯ (edit_date), тож
available_at = edit_date для відредагованих постів — без припущень і без відкидання постів.

Запуск ТІЛЬКИ локально (логін вашим номером телефону):
  1. API-ключі: https://my.telegram.org/apps → «API development tools» → api_id, api_hash
  2. pip install telethon pandas pyarrow
  3. Змінні середовища:
       Windows PowerShell:  $env:TG_API_ID="12345"; $env:TG_API_HASH="abc..."
       Linux/macOS:         export TG_API_ID=12345 TG_API_HASH=abc...
  4. З кореня репозиторію:
       python -m src.telegram_scraper kpszsu --since 2025-10-25
     Перший запуск питає телефон і код з Telegram (вводите у СВОЄМУ терміналі).
     Повторний запуск докачує лише нові пости (і добирає старіші до --since).

!!! air_alerts_session.session = повний доступ до акаунта. Він у .gitignore — НЕ комітити, нікому не передавати.

Вихід: data/raw/telegram/<channel>.parquet
  channel, msg_id, date_utc, edit_date_utc, available_at, edited, is_forward, text
Наприкінці друкується статистика редагувань (частка й затримка) — вона показує, наскільки
ризикованим є веб-прев'ю, де часу редагування не видно.

Правила проти leakage (застосовуються в src/text_features.py):
  - фіча в момент t бере лише пости з available_at < t − 5 хв;
  - available_at = edit_date, якщо пост редагували (API віддає лише поточний текст);
  - views НЕ збираємо: це лічильник на момент скрапу, а не на момент t;
  - видалені пости відсутні (survivorship bias);
  - для forward беремо дату репосту в цьому каналі (m.date), не оригіналу.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/raw/telegram")
COLUMNS = ["channel", "msg_id", "date_utc", "edit_date_utc", "available_at", "edited", "is_forward", "text"]


def message_to_row(m, channel: str) -> dict | None:
    """Telethon Message → рядок. None для службових/порожніх (медіа без підпису)."""
    text = getattr(m, "message", None)
    if not text:
        return None
    return {
        "channel": channel,
        "msg_id": m.id,
        "date_utc": m.date,                       # tz-aware UTC
        "edit_date_utc": m.edit_date,
        "is_forward": m.fwd_from is not None,
        "text": text,
    }


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in ("date_utc", "edit_date_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["edited"] = df["edit_date_utc"].notna()
    df["available_at"] = df["edit_date_utc"].fillna(df["date_utc"])   # поточний текст доступний з моменту редагування
    return df[COLUMNS]


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Свіжо завантажені версії постів мають пріоритет (могли бути відредаговані)."""
    df = pd.concat([x for x in (old, new) if len(x)], ignore_index=True) if len(old) or len(new) else new
    df = df.drop_duplicates("msg_id", keep="last")
    return df.sort_values("date_utc").reset_index(drop=True)


def edit_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "порожньо"
    ed = df[df["edited"]]
    if ed.empty:
        return f"{len(df)} постів, редагувань немає"
    delay = (ed["edit_date_utc"] - ed["date_utc"]).dt.total_seconds() / 60
    return (f"{len(df)} постів; відредаговано {len(ed)} ({len(ed) / len(df):.1%}); затримка редагування, хв: "
            f"медіана {delay.median():.1f}, 90% {delay.quantile(0.9):.1f}, >5 хв — {(delay > 5).mean():.1%}")


async def fetch(client, channel: str, since: pd.Timestamp, old: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(old):
        # 1) нові пости (id > max збереженого), від найновіших униз
        async for m in client.iter_messages(channel, min_id=int(old["msg_id"].max()), wait_time=1):
            if (r := message_to_row(m, channel)):
                rows.append(r)
        # 2) старіші за найстаріший збережений — якщо --since раніше, ніж уже зібрано
        if old["date_utc"].min() > since:
            async for m in client.iter_messages(channel, offset_id=int(old["msg_id"].min()), wait_time=1):
                if m.date < since:
                    break
                if (r := message_to_row(m, channel)):
                    rows.append(r)
    else:
        async for m in client.iter_messages(channel, wait_time=1):   # від найновіших до `since`
            if m.date < since:
                break
            if (r := message_to_row(m, channel)):
                rows.append(r)
            if len(rows) % 5000 == 0 and rows:
                print(f"  {channel}: {len(rows)} … {m.date:%Y-%m-%d %H:%M}", flush=True)
    return merge(old, to_frame(rows))


async def main(channels: list[str], since: str, session: str) -> None:
    from telethon import TelegramClient     # імпорт тут: модуль можна тестувати без telethon

    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    since_ts = pd.Timestamp(since, tz="UTC")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # flood_sleep_threshold: автоматично чекати FloodWait до 5 хв замість падіння
    async with TelegramClient(session, api_id, api_hash, flood_sleep_threshold=300) as client:
        for ch in channels:
            ch = ch.lstrip("@")
            path = OUT_DIR / f"{ch}.parquet"
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
            try:
                df = await fetch(client, ch, since_ts, old)
            except Exception as e:   # неіснуючий/приватний хендл не валить решту каналів
                print(f"{ch}: ПРОПУЩЕНО ({type(e).__name__}: {e})")
                continue
            df.to_parquet(path, index=False)
            span = (f"{df['date_utc'].min():%Y-%m-%d} — {df['date_utc'].max():%Y-%m-%d}" if len(df) else "")
            print(f"{ch}: {span} -> {path}\n  {edit_report(df)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("channels", nargs="+", help="username каналів (з @ або без)")
    p.add_argument("--since", default="2025-10-25", help="UTC-дата, з якої збирати")
    p.add_argument("--session", default="air_alerts_session", help="файл сесії Telethon (не комітити!)")
    a = p.parse_args()
    asyncio.run(main(a.channels, a.since, a.session))
