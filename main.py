import asyncio
import html
import http.server
import json
import logging
import os
import random
import re
import threading
import numpy as np
import openai
from dotenv import load_dotenv
from telegram import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
import analytics

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

ADMIN_IDENTIFIERS = [
    x.strip().lstrip("@").lower()
    for x in os.getenv("ADMIN_USER_ID", os.getenv("ADMIN_USER_IDS", "")).split(",")
    if x.strip()
]


def is_admin(user) -> bool:
    """Check if the user matches the configured ADMIN_USER_ID in env."""
    if not user or not ADMIN_IDENTIFIERS:
        return False
    if str(user.id) in ADMIN_IDENTIFIERS:
        return True
    if user.username and user.username.lstrip("@").lower() in ADMIN_IDENTIFIERS:
        return True
    return False


if not TELEGRAM_BOT_TOKEN:
    logging.warning("TELEGRAM_BOT_TOKEN is not set. The bot will fail to start.")
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY is not set. Embeddings generation will fail.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(__file__)
COMBINED_PATH = os.path.join(BASE_DIR, "utilities", "combined_channel_messages.jsonl")
PRIMARY_PATH = os.path.join(BASE_DIR, "utilities", "channel_messages.jsonl")
MESSAGES_PATH = COMBINED_PATH if os.path.exists(COMBINED_PATH) else PRIMARY_PATH
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "utilities", "channel_embeddings.npy")


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


MESSAGES = load_jsonl(MESSAGES_PATH)
NORMED_MESSAGE_EMBEDDINGS = None


def get_embeddings(texts: list) -> np.ndarray:
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model="text-embedding-3-small")
        all_embeddings.extend([item.embedding for item in response.data])
    return np.array(all_embeddings)


def get_or_load_embeddings() -> np.ndarray:
    """Load precomputed embeddings from disk if available, otherwise compute and cache them."""
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        logging.info(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}...")
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        if len(embeddings) == len(MESSAGES):
            logging.info(f"Loaded {len(embeddings)} cached embeddings in milliseconds.")
            return embeddings
        logging.warning("Embeddings cache size mismatch with channel_messages.jsonl; re-computing...")

    logging.info("Computing channel message embeddings from OpenAI API...")
    embeddings = get_embeddings([m["text"] for m in MESSAGES])
    try:
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
        logging.info(f"Saved embeddings cache to {EMBEDDINGS_CACHE_PATH}.")
    except Exception as e:
        logging.warning(f"Could not save embeddings cache: {e}")
    return embeddings


def rank_messages(user_query: str, seen_ids: set = None) -> list:
    """Compute query embedding and rank messages with freshness deduplication and diversity."""
    query_emb = get_embeddings([user_query])
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    scores = np.dot(NORMED_MESSAGE_EMBEDDINGS, query_norm.T).flatten()

    results = []
    for msg, score in zip(MESSAGES, scores):
        adj_score = float(score)
        msg_id = msg.get("message_id")
        # Penalize already seen messages to surface fresh sermons on repeated queries
        if seen_ids and msg_id in seen_ids:
            adj_score -= 0.15
        results.append((msg, adj_score))

    results.sort(key=lambda x: x[1], reverse=True)

    # Slight stochastic variety among top candidates for natural variety across searches
    top_candidates = results[:12]
    rest = results[12:]
    if len(top_candidates) > 3:
        varied_pool = []
        for msg, sc in top_candidates:
            varied_pool.append((msg, sc + random.uniform(-0.012, 0.012)))
        varied_pool.sort(key=lambda x: x[1], reverse=True)
        return varied_pool + rest

    return results


WELCOME = (
    "Welcome to Pastor Tara Akinkuade's Sermon Assistant.\n\n"
    "This bot helps you discover and listen to relevant audio messages and prophetic teachings directly from Pastor Tara's official channels.\n\n"
    "How to use:\n"
    "• Type any topic, spiritual question, or life situation you are seeking guidance on.\n"
    "• You can play the audio message directly in this chat or tap 'Open in Channel' to view it in the original channel.\n\n"
    "Sample searches:\n"
    "• Building structures for financial prosperity\n"
    "• Stretching in prayer and spiritual encounters\n"
    "• Principles for a fruitful marriage\n"
    "• Destiny realignment and vision\n\n"
    "Top commands:\n"
    "/start - Welcome guide and introduction\n"
    "/categories - Browse sermons by ministry theme\n"
    "/more - Deliver additional messages from your last search\n"
    "/help - Search instructions and tips\n\n"
    "Type your topic or question below to begin."
)

HELP_TEXT = (
    "Pastor Tara's Sermon Assistant — Commands & Guide\n\n"
    "Commands:\n"
    "/start - Welcome guide and introduction\n"
    "/categories - Browse curated ministry themes\n"
    "/more - Deliver additional messages from your last search\n"
    "/help - Show this guide\n\n"
    "Search Tips:\n"
    "• Natural language works best. Rather than just a single word, describe what you are looking for (e.g., 'how to conquer fear and anxiety' or 'wisdom for expanding my business').\n"
    "• Messages are playable directly inside this chat.\n"
    "• Type 'more' anytime after receiving recommendations to view the next set."
)

CATEGORIES_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("Business & Finances", callback_data="cat:Finances & Business"),
        InlineKeyboardButton("Marriage & Family", callback_data="cat:Marriage & Family"),
    ],
    [
        InlineKeyboardButton("Healing & Wellbeing", callback_data="cat:Healing & Health"),
        InlineKeyboardButton("Prayer & Faith", callback_data="cat:Faith & Prayer"),
    ],
    [
        InlineKeyboardButton("Purpose & Destiny", callback_data="cat:Purpose & Destiny"),
    ]
])


async def post_init(application: Application):
    """Register top commands with Telegram menu."""
    commands = [
        BotCommand("start", "Welcome guide and introduction"),
        BotCommand("categories", "Browse sermons by ministry theme"),
        BotCommand("more", "Deliver additional messages from recent search"),
        BotCommand("help", "Search instructions and tips"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except Exception as e:
        logging.warning(f"Could not register Telegram commands menu: {e}")


async def start_command(update: Update, context: CallbackContext) -> None:
    context.user_data["greeted"] = True
    user = update.effective_user
    if user:
        analytics.record_query(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            query_text="/start",
            message_ids=[],
        )
    await update.message.reply_text(WELCOME)


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(HELP_TEXT)


async def categories_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(
        "Select a ministry theme to explore sermons:",
        reply_markup=CATEGORIES_KEYBOARD,
    )


async def deliver_sermons(
    chat_id: int,
    context: CallbackContext,
    ranked_list: list,
    start_idx: int,
    count: int = 3,
    query_text: str = "",
    user=None,
):
    """Deliver a batch of playable audio messages directly to the chat."""
    batch = ranked_list[start_idx : start_idx + count]
    if not batch:
        await context.bot.send_message(
            chat_id=chat_id,
            text="You have reached the end of the available messages on this topic. Type a new topic to explore further.",
        )
        return

    seen_ids = context.user_data.setdefault("seen_ids", set())
    delivered_ids = []

    for i, (msg, _) in enumerate(batch, start_idx + 1):
        text = msg.get("text", "")
        snippet = text[:280].rsplit(" ", 1)[0] + "..." if len(text) > 280 else text
        safe_snippet = html.escape(snippet)
        link = msg.get("channel_link", "")
        audio_message_id = msg.get("message_id")
        from_channel = msg.get("from_chat_id") or (f"@{msg['channel']}" if msg.get("channel") else "@pst_tara")
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Open in Channel", url=link)]]) if link else None
        caption = f"<b>{i}.</b> {safe_snippet}"

        if audio_message_id:
            delivered_ids.append(audio_message_id)
            seen_ids.add(audio_message_id)

        try:
            await context.bot.copy_message(
                chat_id=chat_id,
                from_chat_id=from_channel,
                message_id=audio_message_id,
                caption=caption,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        except Exception:
            logging.exception(f"Failed to copy audio for message {audio_message_id}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=caption,
                parse_mode="HTML",
                reply_markup=keyboard,
                disable_web_page_preview=True,
            )

    new_offset = start_idx + len(batch)
    context.user_data["offset"] = new_offset

    # Show pagination button if more results remain
    if new_offset < len(ranked_list):
        more_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("More Messages", callback_data="action:more")]
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Showing {new_offset} of {len(ranked_list)} relevant messages. Tap below or type 'more' for the next batch.",
            reply_markup=more_keyboard,
        )

    # Record analytics
    if user and query_text:
        analytics.record_query(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            query_text=query_text,
            message_ids=delivered_ids,
        )



async def more_command(update: Update, context: CallbackContext) -> None:
    """Deliver next batch of messages from previous search."""
    last_ranked = context.user_data.get("last_ranked")
    if not last_ranked:
        await update.message.reply_text(
            "You have not searched for a topic yet. Type a topic or question to receive recommendations."
        )
        return

    offset = context.user_data.get("offset", 0)
    query_text = context.user_data.get("last_query", "")
    await deliver_sermons(
        chat_id=update.effective_chat.id,
        context=context,
        ranked_list=last_ranked,
        start_idx=offset,
        count=3,
        query_text=query_text,
        user=update.effective_user,
    )


async def stats_command(update: Update, context: CallbackContext) -> None:
    """Admin-only command to view ministry metrics and visitor needs."""
    user = update.effective_user
    if not is_admin(user):
        # Silently ignore for non-admins so unauthorized users cannot access it
        return

    summary = analytics.get_analytics_summary()
    total_users = summary["total_users"]
    total_queries = summary["total_queries"]
    active_7d = summary["active_7d"]
    categories = summary["category_counts"]
    recent = summary.get("recent_queries", [])

    cat_lines = []
    for cat, count in categories:
        pct = (count / total_queries * 100) if total_queries > 0 else 0
        cat_lines.append(f"• {cat}: {count} ({pct:.0f}%)")
    cat_text = "\n".join(cat_lines) if cat_lines else "• No query data recorded yet."

    recent_lines = []
    for q_text, cat, dt in recent:
        clean_q = q_text[:50].replace("\n", " ")
        recent_lines.append(f"• \"{clean_q}\" ({cat})")
    recent_text = "\n".join(recent_lines) if recent_lines else "• None yet."

    stats_msg = (
        "Ministry Bot Analytics Report\n\n"
        f"Visitors & Activity:\n"
        f"• Total Unique Visitors: {total_users}\n"
        f"• Active Last 7 Days: {active_7d}\n"
        f"• Total Questions Asked: {total_queries}\n\n"
        f"Spiritual Needs Breakdown:\n"
        f"{cat_text}\n\n"
        f"Recent Searches:\n"
        f"{recent_text}\n\n"
        "Data recorded in analytics database."
    )
    await update.message.reply_text(stats_msg)


async def myid_command(update: Update, context: CallbackContext) -> None:
    """Help user check their Telegram User ID and admin status."""
    user = update.effective_user
    username_str = f"@{user.username}" if user.username else "No username set"
    admin_status = "Admin (Authorized)" if is_admin(user) else "Standard User"
    msg = (
        f"Your Telegram Profile:\n"
        f"• User ID: <code>{user.id}</code>\n"
        f"• Username: {username_str}\n"
        f"• Status: {admin_status}\n\n"
        f"To grant admin access, set:\n"
        f"ADMIN_USER_ID={user.id}\n"
        f"in your environment variables."
    )
    await update.message.reply_text(msg, parse_mode="HTML")




async def on_callback_query(update: Update, context: CallbackContext) -> None:
    """Handle inline button clicks for categories and pagination."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data or ""
    if data == "action:more":
        last_ranked = context.user_data.get("last_ranked")
        if not last_ranked:
            await query.message.reply_text("Type a topic to start a new search.")
            return
        offset = context.user_data.get("offset", 0)
        await deliver_sermons(
            chat_id=update.effective_chat.id,
            context=context,
            ranked_list=last_ranked,
            start_idx=offset,
            count=3,
            query_text=context.user_data.get("last_query", ""),
            user=update.effective_user,
        )
    elif data.startswith("cat:"):
        cat_name = data.split("cat:", 1)[1]
        await query.message.reply_text(f"Searching for messages in {cat_name}...")
        seen_ids = context.user_data.get("seen_ids", set())
        ranked = await asyncio.to_thread(rank_messages, cat_name, seen_ids)
        context.user_data["last_ranked"] = ranked
        context.user_data["last_query"] = cat_name
        await deliver_sermons(
            chat_id=update.effective_chat.id,
            context=context,
            ranked_list=ranked,
            start_idx=0,
            count=3,
            query_text=cat_name,
            user=update.effective_user,
        )


def is_more_request(text: str) -> tuple[bool, int]:
    """Check if a user is asking for more messages and parse requested count."""
    clean = text.lower().strip()
    if clean in ["more", "show more", "give me more", "next", "more messages", "continue"]:
        return True, 3
    # Matches patterns like "5 more", "10 more", "give me 5 more"
    match = re.search(r"\b(\d+)\s+more\b", clean)
    if match:
        count = min(int(match.group(1)), 10)
        return True, count
    return False, 3


async def handle_message(update: Update, context: CallbackContext):
    user_query = update.message.text or ""

    # First-time user onboarding
    if not context.user_data.get("greeted"):
        context.user_data["greeted"] = True
        user = update.effective_user
        if user:
            analytics.record_query(
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
                query_text="/start",
                message_ids=[],
            )
        await update.message.reply_text(WELCOME)
        return

    # Check if this is a pagination request
    is_more, count = is_more_request(user_query)
    if is_more:
        last_ranked = context.user_data.get("last_ranked")
        if not last_ranked:
            await update.message.reply_text(
                "You have not searched for a topic yet. Type a topic or question to receive recommendations."
            )
            return
        offset = context.user_data.get("offset", 0)
        await deliver_sermons(
            chat_id=update.effective_chat.id,
            context=context,
            ranked_list=last_ranked,
            start_idx=offset,
            count=count,
            query_text=context.user_data.get("last_query", ""),
            user=update.effective_user,
        )
        return

    await update.message.reply_text("Searching for relevant messages...")

    try:
        seen_ids = context.user_data.get("seen_ids", set())
        # Offload synchronous OpenAI call and ranking with freshness deduplication
        ranked = await asyncio.to_thread(rank_messages, user_query, seen_ids)
    except Exception as e:
        logging.exception("rank_messages failed")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")
        return

    context.user_data["last_ranked"] = ranked
    context.user_data["last_query"] = user_query

    await deliver_sermons(
        chat_id=update.effective_chat.id,
        context=context,
        ranked_list=ranked,
        start_idx=0,
        count=3,
        query_text=user_query,
        user=update.effective_user,
    )


def main():
    global NORMED_MESSAGE_EMBEDDINGS
    raw_embeddings = get_or_load_embeddings()
    # Pre-normalize message embeddings once to make cosine similarity a fast dot product
    NORMED_MESSAGE_EMBEDDINGS = raw_embeddings / (
        np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-10
    )
    logging.info(f"Ready — {len(MESSAGES)} messages indexed.")

    # Render Web Service port binding check
    port = os.getenv("PORT")
    if port:
        class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK")

            def log_message(self, format, *args):
                pass

        def run_health_server():
            server = http.server.HTTPServer(("0.0.0.0", int(port)), HealthCheckHandler)
            server.serve_forever()

        threading.Thread(target=run_health_server, daemon=True).start()
        logging.info(f"Health-check server running on port {port}")

    # In Python 3.12/3.14+, get_event_loop() no longer auto-creates a loop
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("categories", categories_command))
    application.add_handler(CommandHandler("more", more_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("myid", myid_command))
    application.add_handler(CommandHandler("id", myid_command))
    application.add_handler(CallbackQueryHandler(on_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
