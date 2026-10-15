import asyncio
import html
import http.server
import json
import logging
import os
import threading
import numpy as np
import openai
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackContext, CommandHandler, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

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


def rank_messages(user_query: str) -> list:
    """Compute query embedding and rank messages via normalized cosine similarity."""
    query_emb = get_embeddings([user_query])
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    scores = np.dot(NORMED_MESSAGE_EMBEDDINGS, query_norm.T).flatten()
    return sorted(zip(MESSAGES, scores), key=lambda x: x[1], reverse=True)


WELCOME = (
    "Hello! I am Pastor Tara's AI Bot. I am here to assist you in finding messages tailored to your specific needs.\n\n"
    "Simply share a topic you would like to learn about or an area where you are seeking guidance from God's Word, "
    "and I will recommend messages that may be helpful."
)


async def start(update: Update, context: CallbackContext) -> None:
    context.user_data["greeted"] = True
    await update.message.reply_text(WELCOME)


async def handle_message(update: Update, context: CallbackContext):
    user_query = update.message.text

    if not context.user_data.get("greeted"):
        context.user_data["greeted"] = True
        await update.message.reply_text(WELCOME)
        return

    await update.message.reply_text("Searching for relevant messages...")

    try:
        # Offload synchronous OpenAI call and ranking to thread to avoid blocking asyncio loop
        ranked = await asyncio.to_thread(rank_messages, user_query)
    except Exception as e:
        logging.exception("rank_messages failed")
        await update.message.reply_text(f"Sorry, something went wrong: {e}")
        return

    for i, (msg, _) in enumerate(ranked[:3], 1):
        text = msg.get("text", "")
        snippet = text[:280].rsplit(" ", 1)[0] + "..." if len(text) > 280 else text
        safe_snippet = html.escape(snippet)
        link = msg.get("channel_link", "")
        audio_message_id = msg.get("message_id")
        from_channel = msg.get("from_chat_id") or (f"@{msg['channel']}" if msg.get("channel") else "@pst_tara")
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Open in Channel", url=link)]]) if link else None
        caption = f"<b>{i}.</b> {safe_snippet}"

        # Send the audio file with caption and button attached so the user can play it directly
        try:
            await context.bot.copy_message(
                chat_id=update.effective_chat.id,
                from_chat_id=from_channel,
                message_id=audio_message_id,
                caption=caption,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        except Exception:
            logging.exception(f"Failed to send audio for result {i}")
            await update.message.reply_text(
                caption,
                parse_mode="HTML",
                reply_markup=keyboard,
                disable_web_page_preview=True,
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

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
