import os
import logging
import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

# ============================================================
#  LOAD ENV
# ============================================================
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("8992064482:AAFTDUP3SD58fehc0Kpz0jcyGq919dozD0g")
FREEMODEL_API_KEY = os.getenv("fe_oa_9dcaf183ce4139e607d5b2cc7aeda3e628ddeaf71adf4f81")

FREEMODEL_BASE_URL = "https://api.freemodel.dev/v1"
MODEL_NAME = "gpt-5.5"

# ============================================================
#  LOGGING
# ============================================================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================================
#  SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
Kamu adalah Adit Agent Serbaguna 5.0.
Kamu adalah asisten AI pribadi milik Adit.

Kemampuan utama:
- Menjawab pertanyaan umum
- Membantu coding
- Membantu Linux
- Membantu crypto dan blockchain
- Membantu menulis artikel
- Membantu membuat thread Twitter/X
- Brainstorming ide dan strategi

Kepribadian:
- Santai
- Ramah
- Cerdas
- Tidak bertele-tele
- Menjelaskan hal rumit dengan bahasa sederhana

Aturan:
- Gunakan bahasa yang sama dengan pengguna.
- Jangan mengaku sebagai ChatGPT, OpenAI, atau model lain.
- Jika ditanya siapa kamu, jawab: Adit Agent Serbaguna 5.0.
"""

# ============================================================
#  MEMORY USER
# ============================================================
user_histories: dict[int, list] = {}

def get_history(user_id: int):
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]

def reset_history(user_id: int):
    user_histories[user_id] = []

# ============================================================
#  AI CALL (ASYNC)
# ============================================================
async def chat_with_ai(user_id: int, user_message: str) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{FREEMODEL_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {FREEMODEL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                },
            )

        data = response.json()

        if response.status_code != 200:
            logger.error(f"API error: {data}")
            return f"⚠️ API error {response.status_code}"

        reply = data["choices"][0]["message"]["content"]

        history.append({"role": "assistant", "content": reply})

        # keep last 40 messages only
        user_histories[user_id] = history[-40:]

        return reply

    except Exception as e:
        logger.error(f"Request error: {e}")
        return f"⚠️ Error: {e}"

# ============================================================
#  COMMANDS
# ============================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "bro"

    await update.message.reply_text(
        f"Yo {name}! Gue Adit Agent Serbaguna 5.0 🤖\n\n"
        "Gue bisa bantu:\n"
        "- Coding & Linux\n"
        "- Crypto & blockchain\n"
        "- Nulis & brainstorming\n\n"
        "Langsung aja chat. /reset buat mulai ulang.",
    )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_history(update.effective_user.id)
    await update.message.reply_text("🔄 Memory di-reset.")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - mulai bot\n"
        "/reset - reset memory\n"
        "/help - bantuan",
    )

# ============================================================
#  MESSAGE HANDLER
# ============================================================
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    reply = await chat_with_ai(user_id, user_text)

    await update.message.reply_text(reply)

# ============================================================
#  MAIN
# ============================================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN belum di-set di environment")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Adit Agent jalan...")
    app.run_polling()

if __name__ == "__main__":
    main()
