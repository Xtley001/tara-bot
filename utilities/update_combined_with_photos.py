import json
import os
import subprocess
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_PATH = os.path.join(BASE_DIR, "utilities", "combined_channel_messages.jsonl")
PST_TARA_PATH = os.path.join(BASE_DIR, "utilities", "channel_messages.jsonl")
STUDENT_MAP_PATH = os.path.join(BASE_DIR, "utilities", "student_photo_map.json")

# 1. Load pst_tara photo IDs
pst_tara_photos = {}
with open(PST_TARA_PATH, encoding="utf-8") as f:
    for line in f:
        m = json.loads(line)
        if m.get("photo_message_id"):
            pst_tara_photos[m["message_id"]] = m["photo_message_id"]

# 2. Load git business and family flyer mappings
def get_git_channel_photos(file_path):
    raw = subprocess.check_output(["git", "show", f"615b72c~1:{file_path}"], text=True, encoding="utf-8")
    mapping = {}
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        item = json.loads(line)
        audios = item.get("audios", [])
        media = item.get("media", "")
        if audios and media and media not in audios:
            m_photo = re.search(r"/(\d+)$", media)
            photo_id = int(m_photo.group(1)) if m_photo else None
            for a in audios:
                m_audio = re.search(r"/(\d+)$", a)
                if m_audio and photo_id:
                    mapping[int(m_audio.group(1))] = photo_id
    return mapping

biz_photos = get_git_channel_photos("data/business.jsonl")
fam_photos = get_git_channel_photos("data/family.jsonl")

# 3. Load student photos
with open(STUDENT_MAP_PATH, encoding="utf-8") as f:
    student_photos_raw = json.load(f)
student_photos = {int(k): int(v) for k, v in student_photos_raw.items()}

# 4. Read combined_channel_messages.jsonl and inject photo_message_id
with open(COMBINED_PATH, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

updated_count = 0
for r in records:
    ch = r.get("channel")
    mid = r.get("message_id")
    photo_id = None
    if ch == "pst_tara":
        photo_id = pst_tara_photos.get(mid)
    elif ch == "TheSupernaturalBusinessMan":
        photo_id = biz_photos.get(mid)
    elif ch == "TheSupernaturalFamily":
        photo_id = fam_photos.get(mid)
    elif ch == "TheSupernaturalStudent":
        photo_id = student_photos.get(mid)

    if photo_id:
        r["photo_message_id"] = photo_id
        updated_count += 1
    else:
        r["photo_message_id"] = None

print(f"Total records in combined: {len(records)}, Injected photo_message_id: {updated_count}")

# 5. Overwrite combined_channel_messages.jsonl
with open(COMBINED_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Successfully updated combined_channel_messages.jsonl!")
