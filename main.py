import asyncio
import html
import http.server
import json
import logging
import os
import random
import re
import threading
import time
import urllib.request
import numpy as np
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
import discipleship
import series_manager

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
series_manager.init_series_engine(MESSAGES)


def openai_api_request(endpoint: str, payload: dict, timeout: int = 15) -> dict:
    url = f"https://api.openai.com/v1/{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "TaraBot/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_embeddings(texts: list) -> np.ndarray:
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        res = openai_api_request("embeddings", {"input": batch, "model": "text-embedding-3-small"})
        all_embeddings.extend([item["embedding"] for item in res["data"]])
    return np.array(all_embeddings)


def synthesize_pastoral_guidance(user_name: str, query: str, top_sermon: dict) -> str:
    """Generate warm, faith-filled pastoral counsel in Pastor Tara's voice."""
    if not OPENAI_API_KEY:
        return ""
    try:
        sermon_title = (top_sermon.get("audio_filename", "") or "").replace(".mp3", "") or top_sermon.get("text", "")[:60]
        sermon_snippet = top_sermon.get("text", "")[:350]

        system_prompt = (
            "You are the personal AI Ministry Assistant for Pastor Tara Akinkuade (Senior Pastor of The Standing Church). "
            "Speak in Pastor Tara's authentic, faith-filled, warm pastoral voice. Embody his key revelations: "
            "the Blessing of Dominion, stretching in prayer and spiritual encounters, giving credence to God's Word over sensory facts, "
            "spiritual structures for enterprise, and walking in supernatural wisdom.\n\n"
            "Format your response in exactly two short paragraphs:\n"
            "Paragraph 1: Address the believer warmly by their first name (e.g., 'Dear Christley,' or 'Grace and peace to you, Christley.'). "
            "Give a 2-sentence spiritual anchor that directly addresses their inquiry or topic with prophetic clarity.\n"
            "Paragraph 2: A 1-sentence declaration starting with '<b>Confession for Today:</b> ' with the confession inside quotation marks."
        )

        user_content = (
            f"Believer First Name: {user_name}\n"
            f"Spiritual Inquiry / Topic: {query}\n"
            f"Key Sermon Context: {sermon_title}\n"
            f"Excerpt: {sermon_snippet}"
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 220,
            "temperature": 0.6,
        }

        res = openai_api_request("chat/completions", payload, timeout=6)
        content = res["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        logging.warning(f"Could not synthesize pastoral guidance: {e}")
        return ""


def get_sermon_title(msg: dict) -> str:
    """Extract clean sermon title from message metadata, filename, or text."""
    if not msg:
        return "Prophetic Teaching"
    title = msg.get("title", "")
    if title and not title.lower().startswith("message_") and not title.lower().startswith("audio"):
        clean = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", title, flags=re.I).strip()
        clean = re.sub(r"[*_`#]", "", clean).strip()
        if len(clean) > 2:
            return clean
    if msg.get("audio_filename"):
        fn = msg["audio_filename"]
        clean = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", fn, flags=re.I)
        clean = clean.replace("-", " ").strip()
        clean = re.sub(r"[*_`#]", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) > 2 and not clean.lower().startswith("message_") and not clean.lower().startswith("audio"):
            return clean
    text = msg.get("text", "")
    match = re.search(r"[“\"']([^“”\"'\n]{3,60})[”\"']", text)
    if match and not match.group(1).lower().startswith("in this"):
        return match.group(1).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        first = re.sub(r"[*_`#]", "", lines[0]).strip()
        if 3 <= len(first) <= 65 and not first.endswith("."):
            return first
    return "Prophetic Teaching"


def clean_sermon_snippet(raw_text: str, title: str, max_chars: int = 240) -> str:
    """
    Cleans message text for Telegram captions by:
    1. Fixing character encoding artifacts
    2. Removing trailing channel boilerplate
    3. Stripping leading duplicate titles so they are not repeated under the header
    4. Removing raw markdown asterisks, hashes, and backticks
    5. Cleanly truncating to max_chars without cutting mid-word
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\ufffd", "'").replace("â€™", "'").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", " — ").replace("–", " - ")

    # Strip channel boilerplate
    boilerplate_patterns = [
        r"(?:Download|Listen)[\.\s,]+(?:Share[\.\s,]+)?(?:Be\s+Transformed[\.\s,]*)?.*",
        r"Invite people to join this Telegram channel.*",
        r"To be a part of this blessing you are enjoying.*",
        r"For further enquiries.*",
        r"God bless you[\.!]*\s*$",
    ]
    for pat in boilerplate_patterns:
        text = re.sub(pat, "", text, flags=re.I | re.DOTALL).strip()

    # Drop leading line if it duplicates the title header
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        first_clean = re.sub(r"[*_`#\"']", "", lines[0]).strip().lower()
        first_clean = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", first_clean, flags=re.I).strip()
        title_clean = re.sub(r"[*_`#\"']", "", title).strip().lower()
        title_clean = re.sub(r"\.(mp3|m4a|wav|aac|ogg)$", "", title_clean, flags=re.I).strip()
        f_base = re.sub(r"\s+[-–—,]?\s*(?:part|pt|episode|ep)?\s*(\d+|[a-e])\s*$", "", first_clean)
        t_base = re.sub(r"\s+[-–—,]?\s*(?:part|pt|episode|ep)?\s*(\d+|[a-e])\s*$", "", title_clean)

        if f_base == t_base or first_clean == title_clean:
            lines = lines[1:]

    text = " ".join(lines)
    # Strip raw markdown markup
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if len(text) > max_chars:
        cut = text[:max_chars]
        last_dot = cut.rfind(".")
        if last_dot > max_chars * 0.65:
            text = cut[:last_dot + 1]
        else:
            text = cut.rsplit(" ", 1)[0] + "..."

    return html.escape(text)


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


def get_welcome_message(user, profile: dict = None) -> tuple[str, InlineKeyboardMarkup | None]:
    name = (user.first_name if user and user.first_name else "").strip() or "Beloved"
    last_topic = profile.get("last_topic") if profile else None

    # Returning discipleship continuity
    if last_topic and profile and profile.get("visit_count", 0) > 1:
        text = (
            f"Welcome back, <b>{name}</b>, to Pastor Tara AI — where we are raising a supernatural army and restoring dignity to all.\n\n"
            f"Last time, you were meditating on <i>{html.escape(last_topic)}</i>.\n\n"
            "Tell me how I can be of help to you today. "
            "Type what is on your heart, or let me know if you would like to continue in this series."
        )
        return text, None

    text = (
        f"Welcome, <b>{name}</b>, to Pastor Tara AI — where we are raising a supernatural army and restoring dignity to all.\n\n"
        "Tell me how I can be of help to you today. "
        "Simply describe what is on your heart or any area you desire breakthrough in "
        "(e.g., <i>finances, prayer, purpose, marriage, or academics</i>), "
        "and I will share prophetic teachings from Pastor Tara to anchor your spirit."
    )
    return text, None


HELP_TEXT = (
    "<b>Pastor Tara AI — Commands & Guide</b>\n\n"
    "<b>Commands:</b>\n"
    "/start - Welcome introduction and guide\n"
    "/more - Deliver next batch of 5 teachings\n"
    "/help - Show this guide\n\n"
    "<b>How to use:</b>\n"
    "• Simply speak in your own words. Describe what you are walking through or desiring direction in (e.g., 'wisdom for expanding my business' or 'how to conquer fear in prayer').\n"
    "• Each message includes the sermon flyer, description, and playable audio.\n"
    "• Type 'more' or 'next' anytime to receive the next batch of teachings."
)


async def post_init(application: Application):
    """Register top commands with Telegram menu."""
    commands = [
        BotCommand("start", "Welcome introduction and guide"),
        BotCommand("more", "Deliver next batch of 5 teachings"),
        BotCommand("help", "Instructions and search tips"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except Exception as e:
        logging.warning(f"Could not register Telegram commands menu: {e}")


async def start_command(update: Update, context: CallbackContext) -> None:
    context.user_data["greeted"] = True
    user = update.effective_user
    profile = {}
    if user:
        profile = discipleship.get_user_profile(user.id)
        discipleship.record_user_visit_and_topic(user.id, user.first_name, user.username)
        analytics.record_query(
            user_id=user.id,
            username=user.username,
            first_name=user.first_name,
            query_text="/start",
            message_ids=[],
        )
    welcome_text, reply_markup = get_welcome_message(user, profile)
    await update.message.reply_text(welcome_text, parse_mode="HTML", reply_markup=reply_markup)


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="HTML")


async def deliver_sermons(
    chat_id: int,
    context: CallbackContext,
    ranked_list: list,
    start_idx: int,
    count: int = 5,
    query_text: str = "",
    user=None,
):
    """Deliver pastoral guidance and a batch of playable audio messages directly to the chat."""
    if not ranked_list or start_idx >= len(ranked_list):
        await context.bot.send_message(
            chat_id=chat_id,
            text="You have reached the end of the available messages on this topic. Type another topic or question on your heart to continue.",
        )
        return

    # Select teachings from ranked_list (each element in ranked_list is a list of sermon parts)
    selected_teachings = []
    total_parts = 0
    curr_idx = start_idx

    while curr_idx < len(ranked_list):
        teaching = ranked_list[curr_idx]
        selected_teachings.append(teaching)
        total_parts += len(teaching)
        curr_idx += 1
        if total_parts >= count:
            break

    user_name = (user.first_name if user and user.first_name else "").strip() or "Beloved"

    # On the first batch of a topic, deliver pastoral synthesis & prophetic confession
    if start_idx == 0 and query_text and selected_teachings:
        top_sermon = selected_teachings[0][0]
        try:
            guidance = await asyncio.to_thread(synthesize_pastoral_guidance, user_name, query_text, top_sermon)
            if guidance:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"{guidance}\n\n<i>Here are the prophetic messages to anchor this in your spirit today:</i>",
                    parse_mode="HTML",
                )
        except Exception:
            logging.exception("Error sending pastoral guidance")

    seen_ids = context.user_data.setdefault("seen_ids", set())
    delivered_ids = []
    sermon_counter = context.user_data.get("sermon_counter", 0)

    for teaching in selected_teachings:
        for msg in teaching:
            sermon_counter += 1
            title = msg.get("display_title") or get_sermon_title(msg)
            safe_title = html.escape(title)
            safe_snippet = clean_sermon_snippet(msg.get("text", ""), title)

            link = msg.get("channel_link", "") or msg.get("url", "")
            audio_message_id = msg.get("message_id")
            photo_message_id = msg.get("photo_message_id")
            audio_url = msg.get("audio_url", "")
            from_channel = msg.get("from_chat_id") or (f"@{msg['channel']}" if msg.get("channel") else "@pst_tara")

            if safe_snippet:
                caption = f"<b>{sermon_counter}. {safe_title}</b>\n\n{safe_snippet}"
            else:
                caption = f"<b>{sermon_counter}. {safe_title}</b>\n\nProphetic teaching by Pastor Tara Akinkuade."

            # 1. SEND FLYER IMAGE (ABOVE AUDIO) IF VERIFIED PHOTO EXISTS
            if photo_message_id:
                try:
                    await context.bot.copy_message(
                        chat_id=chat_id,
                        from_chat_id=from_channel,
                        message_id=photo_message_id,
                    )
                except Exception:
                    logging.exception(f"Failed to copy flyer photo {photo_message_id} from {from_channel}")

            # 2. SEND AUDIO MESSAGE WITH TITLE, DESCRIPTION, AND LINK
            if audio_message_id:
                delivered_ids.append(audio_message_id)
                seen_ids.add(audio_message_id)
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Open in Channel", url=link)]]) if link else None

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
            elif audio_url:
                buttons = [[InlineKeyboardButton("🎧 Listen Online", url=audio_url)]]
                if link and link != audio_url:
                    buttons.append([InlineKeyboardButton("Open Sermon Page", url=link)])
                keyboard = InlineKeyboardMarkup(buttons)

                try:
                    await context.bot.send_audio(
                        chat_id=chat_id,
                        audio=audio_url,
                        caption=caption,
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
                except Exception:
                    logging.exception(f"Failed to stream audio URL {audio_url}")
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=caption,
                        parse_mode="HTML",
                        reply_markup=keyboard,
                        disable_web_page_preview=True,
                    )
            else:
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Open in Channel", url=link)]]) if link else None
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=caption,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                    disable_web_page_preview=True,
                )

    context.user_data["sermon_counter"] = sermon_counter
    new_offset = curr_idx
    context.user_data["offset"] = new_offset

    # Show pastoral pagination prompt if more results remain
    if new_offset < len(ranked_list):
        more_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Next 5 Teachings", callback_data="action:more")]
        ])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"{user_name}, meditate on these teachings today. When you are ready for more messages, tap below or type 'more'.",
            reply_markup=more_keyboard,
        )

    # Record analytics and discipleship journey
    if user and query_text:
        discipleship.record_user_visit_and_topic(user.id, user.first_name, user.username, query_text)
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
            "You have not searched for a topic yet. Tell me what is on your heart to receive recommendations."
        )
        return

    offset = context.user_data.get("offset", 0)
    query_text = context.user_data.get("last_query", "")
    await deliver_sermons(
        chat_id=update.effective_chat.id,
        context=context,
        ranked_list=last_ranked,
        start_idx=offset,
        count=5,
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
    """Handle inline button clicks for pagination."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data or ""
    if data == "action:more":
        last_ranked = context.user_data.get("last_ranked")
        if not last_ranked:
            await query.message.reply_text("Tell me what is on your heart to start a new search.")
            return
        offset = context.user_data.get("offset", 0)
        await deliver_sermons(
            chat_id=update.effective_chat.id,
            context=context,
            ranked_list=last_ranked,
            start_idx=offset,
            count=5,
            query_text=context.user_data.get("last_query", ""),
            user=update.effective_user,
        )
    elif data == "action:browse_categories" or data.startswith("cat:"):
        await query.message.reply_text("Tell me what is on your heart today, and I will share prophetic teachings from Pastor Tara.")


def is_more_request(text: str) -> tuple[bool, int]:
    """Check if a user is asking for more messages and parse requested count."""
    clean = text.lower().strip()
    if clean in ["more", "show more", "give me more", "next", "more messages", "continue", "next sermon", "next message", "next teachings", "more teachings"]:
        return True, 5
    # Matches patterns like "5 more", "10 more", "give me 5 more"
    match = re.search(r"\b(\d+)\s+more\b", clean)
    if match:
        count = min(int(match.group(1)), 10)
        return True, max(count, 1)
    return False, 5


async def handle_message(update: Update, context: CallbackContext):
    user_query = update.message.text or ""
    user = update.effective_user
    name = (user.first_name if user and user.first_name else "").strip() or "Beloved"

    # First-time user onboarding
    if not context.user_data.get("greeted"):
        context.user_data["greeted"] = True
        profile = {}
        if user:
            profile = discipleship.get_user_profile(user.id)
            discipleship.record_user_visit_and_topic(user.id, user.first_name, user.username)
            analytics.record_query(
                user_id=user.id,
                username=user.username,
                first_name=user.first_name,
                query_text="/start",
                message_ids=[],
            )
        welcome_text, reply_markup = get_welcome_message(user, profile)
        await update.message.reply_text(welcome_text, parse_mode="HTML", reply_markup=reply_markup)
        return

    # Check if this is a pagination request
    is_more, count = is_more_request(user_query)
    if is_more:
        last_ranked = context.user_data.get("last_ranked")
        if not last_ranked:
            await update.message.reply_text(
                f"{name}, you have not searched for a topic yet. Tell me what is on your heart to begin."
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

    await update.message.reply_text(f"Searching for messages for you, {name}...")

    try:
        seen_ids = context.user_data.get("seen_ids", set())
        # Offload synchronous OpenAI call and ranking with freshness deduplication
        ranked = await asyncio.to_thread(rank_messages, user_query, seen_ids)
        grouped_teachings = series_manager.group_ranked_teachings(ranked)
    except Exception as e:
        logging.exception("rank_messages failed")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")
        return

    context.user_data["last_ranked"] = grouped_teachings
    context.user_data["last_query"] = user_query
    context.user_data["sermon_counter"] = 0

    await deliver_sermons(
        chat_id=update.effective_chat.id,
        context=context,
        ranked_list=grouped_teachings,
        start_idx=0,
        count=5,
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

    # Render Web Service port binding check and keep-alive server
    port = os.getenv("PORT")
    if port:
        class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                payload = json.dumps({
                    "status": "online",
                    "service": "Pastor Tara AI Ministry Assistant",
                    "messages_indexed": len(MESSAGES),
                }).encode("utf-8")
                self.wfile.write(payload)

            def log_message(self, format, *args):
                pass

        def run_health_server():
            server = http.server.HTTPServer(("0.0.0.0", int(port)), HealthCheckHandler)
            server.serve_forever()

        threading.Thread(target=run_health_server, daemon=True).start()
        logging.info(f"Health-check server running on port {port}")

        # Keep-alive daemon thread to prevent Render free-tier from idling after 15 minutes
        render_url = (os.getenv("RENDER_EXTERNAL_URL") or "https://tara-bot-flyg.onrender.com").rstrip("/")
        
        def keep_alive_loop():
            # Wait 90 seconds after boot before sending first self-ping
            time.sleep(90)
            while True:
                try:
                    req = urllib.request.Request(
                        f"{render_url}/",
                        headers={"User-Agent": "TaraBot-KeepAlive/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        logging.info(f"Keep-alive ping sent to {render_url}: HTTP {resp.status}")
                except Exception as e:
                    logging.warning(f"Keep-alive ping failed: {e}")
                # Ping every 9 minutes (Render free tier sleeps after 15 minutes of inactivity)
                time.sleep(540)

        threading.Thread(target=keep_alive_loop, daemon=True).start()
        logging.info(f"Keep-alive daemon active for {render_url} (interval: 9m)")

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
    application.add_handler(CommandHandler("more", more_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("myid", myid_command))
    application.add_handler(CommandHandler("id", myid_command))
    application.add_handler(CallbackQueryHandler(on_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
