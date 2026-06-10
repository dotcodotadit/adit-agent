import logging
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ============================================================
#  KONFIGURASI
# ============================================================
TELEGRAM_BOT_TOKEN = ("8992064482:AAFTDUP3SD58fehc0Kpz0jcyGq919dozD0g")
FREEMODEL_API_KEY  = ("fe_oa_9dcaf183ce4139e607d5b2cc7aeda3e628ddeaf71adf4f81")
FREEMODEL_BASE_URL = "https://api.freemodel.dev/v1"
MODEL_NAME = "gpt-5.5"
# ============================================================

SYSTEM_PROMPT = """Kamu adalah Adit Agent Serbaguna 5.0.
Kamu adalah asisten AI pribadi milik Adit.
Kemampuan utama:
* Menjawab pertanyaan umum
* Membantu coding
* Membantu Linux
* Membantu crypto dan blockchain
* Membantu menulis artikel
* Membantu membuat thread Twitter/X
* Brainstorming ide dan strategi

Kepribadian:
* Santai
* Ramah
* Cerdas
* Tidak bertele-tele
* Menjelaskan hal rumit dengan bahasa sederhana

Aturan:
* Gunakan bahasa yang sama dengan pengguna.
* Jika pengguna memakai bahasa Indonesia, jawab dalam bahasa Indonesia yang natural.
* Jika pengguna memakai bahasa Inggris, jawab dalam bahasa Inggris.
* Jangan mengaku sebagai Codex, ChatGPT, OpenAI, atau model lain kecuali ditanya secara teknis.
* Jika ditanya siapa dirimu, perkenalkan dirimu sebagai Adit Agent Serbaguna 5.0."""

user_histories: dict[int, list] = {}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_history(user_id: int) -> list:
    if user_id not in user_histories:
        user_histories[user_id] = []
    return user_histories[user_id]

def reset_history(user_id: int):
    user_histories[user_id] = []

def chat_with_ai(user_id: int, user_message: str) -> str:
    history = get_history(user_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.post(
                f"{FREEMODEL_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {FREEMODEL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                }
            )
            logger.info(f"Status: {response.status_code}")
            data = response.json()

            if response.status_code != 200:
                logger.error(f"API error response: {data}")
                return f"⚠️ API error {response.status_code}: {data}"

            reply = data["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": reply})

            if len(history) > 40:
                user_histories[user_id] = history[-40:]

            return reply

    except Exception as e:
        logger.error(f"Request error: {e}")
        return f"⚠️ Error: {e}"


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_user.first_name or "bro"
    await update.message.reply_text(
        f"Yo {name}! Gue *Adit Agent Serbaguna 5.0* 🤖\n\n"
        "Gue bisa bantu lo buat:\n"
        "• Pertanyaan umum\n"
        "• Coding & Linux\n"
        "• Crypto & blockchain\n"
        "• Nulis artikel / thread Twitter\n"
        "• Brainstorming ide\n\n"
        "Langsung ketik aja pertanyaan lo. Kalau mau reset obrolan, ketik /reset.",
        parse_mode="Markdown"
    )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_history(update.effective_user.id)
    await update.message.reply_text("🔄 Obrolan udah di-reset. Mulai fresh lagi yuk!")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Command yang tersedia:*\n"
        "/start — Perkenalan bot\n"
        "/reset — Reset history percakapan\n"
        "/help  — Tampilkan pesan ini",
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    reply = chat_with_ai(user_id, user_text)
    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot jalan... tekan Ctrl+C untuk stop.")
    app.run_polling()