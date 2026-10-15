"""
Multi-Channel Scraper for Pastor Tara's Telegram Channels:
  - @pst_tara -> utilities/channel_messages.jsonl
  - @TheSupernaturalBusinessMan -> data/business.jsonl
  - @TheSupernaturalFamily -> data/family.jsonl
  - @TheSupernaturalStudent -> data/student.jsonl

Also builds a combined index for unified search if needed.
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto,
    DocumentAttributeAudio,
    DocumentAttributeFilename,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION_PATH = os.path.join(BASE_DIR, "tara_channels_session")

CHANNELS = [
    {
        "username": "pst_tara",
        "output": os.path.join(BASE_DIR, "utilities", "channel_messages.jsonl"),
    },
    {
        "username": "TheSupernaturalBusinessMan",
        "output": os.path.join(BASE_DIR, "data", "business.jsonl"),
    },
    {
        "username": "TheSupernaturalFamily",
        "output": os.path.join(BASE_DIR, "data", "family.jsonl"),
    },
    {
        "username": "TheSupernaturalStudent",
        "output": os.path.join(BASE_DIR, "data", "student.jsonl"),
    },
]


def is_audio(message) -> bool:
    if not isinstance(message.media, MessageMediaDocument):
        return False
    doc = message.media.document
    for attr in doc.attributes:
        if isinstance(attr, (DocumentAttributeAudio,)):
            return True
        if isinstance(attr, DocumentAttributeFilename):
            fn = attr.file_name.lower()
            if fn.endswith((".mp3", ".m4a", ".ogg", ".wav", ".aac", ".opus")):
                return True
    return False


def get_audio_info(message):
    doc = message.media.document
    file_id = f"{doc.id}:{doc.access_hash}"
    filename = ""
    for attr in doc.attributes:
        if isinstance(attr, DocumentAttributeFilename):
            filename = attr.file_name
    return file_id, filename


def has_photo(message):
    return isinstance(message.media, MessageMediaPhoto)


async def scrape_channel(client: TelegramClient, channel_info: dict):
    username = channel_info["username"]
    out_path = channel_info["output"]
    print(f"\n==========================================")
    print(f"📡 Fetching messages from @{username} ...")
    print(f"==========================================")

    all_msgs = []
    async for msg in client.iter_messages(username, reverse=True):
        all_msgs.append(msg)
    print(f"  Loaded {len(all_msgs)} total messages from @{username}")

    groups: dict[int, list] = {}
    for msg in all_msgs:
        if msg.grouped_id:
            groups.setdefault(msg.grouped_id, []).append(msg)

    posts = []
    seen_ids = set()

    for i, msg in enumerate(all_msgs):
        if msg.id in seen_ids:
            continue

        if not is_audio(msg):
            continue

        audio_file_id, audio_filename = get_audio_info(msg)
        photo_msg = None
        text = msg.message or ""

        # Case 1: grouped album
        if msg.grouped_id and msg.grouped_id in groups:
            siblings = groups[msg.grouped_id]
            for sib in siblings:
                seen_ids.add(sib.id)
                if photo_msg is None and has_photo(sib):
                    photo_msg = sib
                if not text and sib.message:
                    text = sib.message

        # Case 2: next message is photo
        if photo_msg is None and i + 1 < len(all_msgs):
            nxt = all_msgs[i + 1]
            if has_photo(nxt) and not nxt.grouped_id:
                photo_msg = nxt
                seen_ids.add(nxt.id)
                if not text and nxt.message:
                    text = nxt.message

        # Case 3: next message is plain text
        check_idx = i + 1
        while check_idx < len(all_msgs):
            nxt = all_msgs[check_idx]
            if nxt.id in seen_ids:
                check_idx += 1
                continue
            if check_idx == i + 1 and not nxt.media and nxt.message and not is_audio(nxt):
                if not text:
                    text = nxt.message
                seen_ids.add(nxt.id)
            break

        seen_ids.add(msg.id)

        if not text:
            continue

        photo_message_id = photo_msg.id if photo_msg is not None else None

        post = {
            "channel": username,
            "from_chat_id": f"@{username}",
            "message_id": msg.id,
            "text": text.strip(),
            "date": msg.date.strftime("%Y-%m-%d"),
            "channel_link": f"https://t.me/{username}/{msg.id}",
            "photo_message_id": photo_message_id,
            "audio_filename": audio_filename,
        }
        posts.append(post)

    posts.sort(key=lambda p: p["message_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(posts)} audio messages for @{username} -> {out_path}")
    return posts


async def main():
    PHONE = os.environ.get("TELEGRAM_PHONE")
    client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
    await client.start(phone=PHONE)

    all_channel_posts = []
    for ch in CHANNELS:
        try:
            posts = await scrape_channel(client, ch)
            all_channel_posts.extend(posts)
        except Exception as e:
            print(f"❌ Failed to scrape @{ch['username']}: {e}")

    # Optionally write combined messages
    combined_path = os.path.join(BASE_DIR, "utilities", "combined_channel_messages.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for p in all_channel_posts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n🎉 Done! Total {len(all_channel_posts)} posts compiled across all 4 channels.")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
