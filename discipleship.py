import json
import logging
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "discipleship.db")


def init_discipleship_db():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                first_name TEXT,
                username TEXT,
                last_topic TEXT,
                favorite_topics TEXT,
                visit_count INTEGER DEFAULT 1,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.warning(f"Could not initialize discipleship DB: {e}")


def get_user_profile(user_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT first_name, username, last_topic, favorite_topics, visit_count, last_seen
            FROM user_profiles WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            favs = json.loads(row[3]) if row[3] else []
            return {
                "first_name": row[0],
                "username": row[1],
                "last_topic": row[2],
                "favorite_topics": favs,
                "visit_count": row[4],
                "last_seen": row[5],
            }
    except Exception as e:
        logging.debug(f"Error fetching user profile: {e}")
    return {}


def record_user_visit_and_topic(user_id: int, first_name: str, username: str, topic: str = None) -> dict:
    """Record user visit and update their spiritual journey context."""
    now = datetime.utcnow().isoformat()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT last_topic, favorite_topics, visit_count FROM user_profiles WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()

        if row:
            prev_last_topic = row[0]
            favs = json.loads(row[1]) if row[1] else []
            visit_count = (row[2] or 1) + 1
            if topic and topic != "/start" and not topic.startswith("/"):
                prev_last_topic = topic
                if topic not in favs:
                    favs.append(topic)
                if len(favs) > 8:
                    favs.pop(0)

            cursor.execute("""
                UPDATE user_profiles
                SET first_name = ?, username = ?, last_topic = ?, favorite_topics = ?,
                    visit_count = ?, last_seen = ?
                WHERE user_id = ?
            """, (first_name or "", username or "", prev_last_topic, json.dumps(favs), visit_count, now, user_id))
        else:
            prev_last_topic = topic if (topic and not topic.startswith("/")) else None
            favs = [topic] if prev_last_topic else []
            cursor.execute("""
                INSERT INTO user_profiles
                (user_id, first_name, username, last_topic, favorite_topics, visit_count, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """, (user_id, first_name or "", username or "", prev_last_topic, json.dumps(favs), now, now))

        conn.commit()
        conn.close()
        return {"last_topic": prev_last_topic, "favorite_topics": favs}
    except Exception as e:
        logging.debug(f"Error recording user topic: {e}")
        return {}


init_discipleship_db()

