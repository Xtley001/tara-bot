import json
import os
import sqlite3
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "analytics.db")

# Predefined categories for spiritual needs tracking
CATEGORY_KEYWORDS = {
    "Finances & Business": [
        "money", "business", "finance", "finances", "wealth", "prosper", "prosperity",
        "debt", "career", "job", "work", "enterprise", "investment", "profit",
        "poverty", "income", "famine", "abundance", "enlargement", "giving", "tithe"
    ],
    "Healing & Health": [
        "heal", "healing", "health", "sickness", "sick", "pain", "disease", "body",
        "hospital", "doctor", "affliction", "infirmity", "weakness", "cure", "restoration"
    ],
    "Marriage & Family": [
        "marriage", "spouse", "husband", "wife", "children", "child", "family", "parent",
        "divorce", "single", "relationship", "wedding", "home", "parenting", "fruitful"
    ],
    "Faith & Prayer": [
        "pray", "prayer", "prayers", "fast", "fasting", "intercession", "faith", "doubt",
        "believe", "trust", "spirit", "holy ghost", "mantle", "tongues", "altar", "incense"
    ],
    "Purpose & Destiny": [
        "purpose", "destiny", "calling", "direction", "vision", "decision", "future",
        "guidance", "lead", "plan", "potential", "identity", "excellence", "exploits"
    ]
}


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            query_count INTEGER DEFAULT 1
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query_text TEXT,
            category TEXT,
            message_ids TEXT,
            created_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def detect_category(query_text: str) -> str:
    text = query_text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return cat
    return "General Spiritual Growth"


def record_query(user_id: int, username: str, first_name: str, query_text: str, message_ids: list):
    try:
        now = datetime.utcnow().isoformat()
        category = detect_category(query_text)
        message_ids_json = json.dumps(message_ids)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Update or insert user
        cursor.execute("""
            INSERT INTO users (user_id, username, first_name, first_seen, last_seen, query_count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_seen = excluded.last_seen,
                query_count = users.query_count + 1
        """, (user_id, username or "", first_name or "", now, now))

        # Insert query log
        cursor.execute("""
            INSERT INTO queries (user_id, query_text, category, message_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, query_text, category, message_ids_json, now))

        conn.commit()
        conn.close()
    except Exception as e:
        # Analytics logging should never crash user experience
        pass


def get_analytics_summary() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries")
    total_queries = cursor.fetchone()[0]

    # Active in last 7 days
    seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    cursor.execute("SELECT COUNT(*) FROM users WHERE last_seen >= ?", (seven_days_ago,))
    active_7d = cursor.fetchone()[0]

    # Category breakdown
    cursor.execute("""
        SELECT category, COUNT(*) as cnt
        FROM queries
        GROUP BY category
        ORDER BY cnt DESC
    """)
    category_counts = cursor.fetchall()

    conn.close()

    return {
        "total_users": total_users,
        "total_queries": total_queries,
        "active_7d": active_7d,
        "category_counts": category_counts,
    }


# Initialize DB upon module load
init_db()
