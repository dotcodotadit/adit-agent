import os
import json
import time
import sqlite3
import logging
import requests
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from concurrent.futures import ThreadPoolExecutor

# ======================
# LOAD ENV
# ======================
load_dotenv(dotenv_path=".env")

# ======================
# LOGGING
# ======================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ======================
# SAFETY CHECK
# ======================
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise Exception("Missing TELEGRAM_BOT_TOKEN in .env")
if not os.getenv("FREEMODEL_API_KEY"):
    raise Exception("Missing FREEMODEL_API_KEY in .env")
if not os.getenv("TAVILY_API_KEY"):
    raise Exception("Missing TAVILY_API_KEY in .env — daftar gratis di https://app.tavily.com")

# ======================
# CONFIG
# ======================
client = OpenAI(
    api_key=os.getenv("FREEMODEL_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.freemodel.dev/v1")
)

# Thread pool untuk DB operations (non-blocking style)
db_executor = ThreadPoolExecutor(max_workers=2)

def test_llm():
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "reply only OK"}],
            max_tokens=5
        )
        logger.info("✅ LLM connection successful")
    except Exception as e:
        logger.error(f"❌ LLM connection failed: {e}")

MAX_HISTORY = 20
RATE_LIMIT_SECONDS = 5
TAVILY_TIMEOUT = 10  # ⭐ NEW: Timeout untuk web search
LLM_TIMEOUT = 30     # ⭐ NEW: Timeout untuk final LLM

user_last_request: dict[int, float] = {}

# ======================
# DATABASE — ASYNC STYLE
# ======================
DB_PATH = os.getenv("DB_PATH", "memory.db")

def init_db():
    """Buat tabel dengan indexing untuk performa"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # ⭐ INDEX untuk query lebih cepat
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_chat_id 
        ON chat_history(chat_id, id DESC)
    """)
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized at {DB_PATH}")

def load_history(chat_id: int) -> list:
    """Load history - optimized dengan executor"""
    def _load():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT role, content FROM chat_history
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (chat_id, MAX_HISTORY))
        rows = c.fetchall()
        conn.close()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    
    return _load()  # Still sync untuk simplicity, but bisa di-async later

def save_message_async(chat_id: int, role: str, content: str):
    """⭐ Save message di background (non-blocking)"""
    def _save():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO chat_history (chat_id, role, content)
            VALUES (?, ?, ?)
        """, (chat_id, role, content))
        conn.commit()
        conn.close()
        
        # Trim di background juga
        trim_history(chat_id)
    
    # Schedule di thread pool (non-blocking)
    db_executor.submit(_save)

def clear_history(chat_id: int):
    """Hapus semua history"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

def trim_history(chat_id: int):
    """Hapus pesan lama"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        DELETE FROM chat_history
        WHERE chat_id = ? AND id NOT IN (
            SELECT id FROM chat_history
            WHERE chat_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
    """, (chat_id, chat_id, MAX_HISTORY))
    conn.commit()
    conn.close()

# ======================
# TOOLS
# ======================

def web_search(query: str) -> str:
    """⭐ Web search dengan timeout"""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": 3,  # ⭐ Reduced dari 5 → lebih cepat
            "search_depth": "basic",
            "include_answer": True
        }
        res = requests.post(url, json=payload, timeout=TAVILY_TIMEOUT)
        res.raise_for_status()
        data = res.json()

        output = []
        if data.get("answer"):
            output.append(f"💡 {data['answer'][:150]}...")
        
        for item in data.get("results", [])[:2]:  # ⭐ Top 2 results only
            title = item.get("title", "")
            url_ = item.get("url", "")
            content = item.get("content", "")
            output.append(f"📌 {title[:100]}\n   {content[:150]}...\n   {url_}")

        return "\n\n".join(output) if output else "No results."

    except requests.exceptions.Timeout:
        logger.warning("Tavily search timeout")
        return "⚠️ Search timeout - coba lagi nanti"
    except requests.exceptions.HTTPError as e:
        logger.error(f"Tavily HTTP error: {e}")
        if e.response.status_code == 401:
            return "⚠️ Tavily API key tidak valid"
        if e.response.status_code == 429:
            return "⚠️ Tavily quota habis"
        return f"Search error: {str(e)}"
    except Exception as e:
        logger.error(f"web_search error: {e}")
        return f"Search failed: {str(e)}"

def write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ File created: {path}"
    except Exception as e:
        logger.error(f"write_file error: {e}", exc_info=True)
        return f"❌ Error writing file: {str(e)}"

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"read_file error: {e}")
        return f"❌ Error reading file: {str(e)}"

# ======================
# SMART AGENT (TANPA PLANNER - DIRECT LLM)
# ======================

def run_agent(user_input: str, chat_id: int) -> str:
    """
    ⭐ OPTIMIZED: Hapus planner step, langsung ke final LLM
    - Planner hanya adds ~2-3 detik tapi agak redundant
    - LLM sudah cukup smart untuk decide tools
    - Kalo perlu tools, bisa pakai function_calling di masa depan
    """
    
    # Load history dari SQLite
    history = load_history(chat_id)

    system_msg = {
        "role": "system",
        "content": (
            "You are Adit Agent, an advanced and helpful AI assistant. "
            "You remember the conversation history with this user. "
            "Answer clearly, helpfully, and in the same language the user uses. "
            "Be concise and direct - answer within 2-3 sentences if possible."
        )
    }

    # Cek apakah perlu web search based on keywords (simple heuristic)
    observations = []
    if any(keyword in user_input.lower() for keyword in ["cari", "search", "google", "berita", "terkini", "hari ini"]):
        logger.info(f"Attempting web search for: {user_input}")
        result = web_search(user_input)
        observations.append(result)

    tool_context = ""
    if observations:
        tool_context = "\n\n[Search Results]\n" + "\n\n".join(observations)

    current_user_msg = {
        "role": "user",
        "content": user_input + tool_context
    }

    messages = [system_msg] + history + [current_user_msg]

    try:
        # ⭐ Timeout + max_tokens untuk response cepat
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300,  # ⭐ NEW: Limit response length
            timeout=LLM_TIMEOUT  # ⭐ NEW: API timeout
        )

        answer = res.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        return "⚠️ AI Provider Error - coba lagi nanti"

    # ⭐ ASYNC: Simpan ke database di background (jangan block response)
    save_message_async(chat_id, "user", user_input)
    save_message_async(chat_id, "assistant", answer)

    return answer

# ======================
# HELPER: kirim pesan panjang
# ======================

async def send_long_message(update: Update, text: str):
    MAX_LEN = 4096
    if len(text) <= MAX_LEN:
        await update.message.reply_text(text)
    else:
        for i in range(0, len(text), MAX_LEN):
            await update.message.reply_text(text[i:i + MAX_LEN])

# ======================
# TELEGRAM HANDLERS
# ======================

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Rate limiting
    now = time.time()
    last = user_last_request.get(chat_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        await update.message.reply_text(
            f"⏳ Tunggu {RATE_LIMIT_SECONDS}s sebelum pesan berikutnya"
        )
        return
    user_last_request[chat_id] = now

    # Typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        response = run_agent(user_text, chat_id)
        await send_long_message(update, response)
    except Exception as e:
        logger.error(f"handle error for chat_id {chat_id}: {e}")
        await update.message.reply_text("❌ Error - coba lagi")

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    history = load_history(chat_id)
    if history:
        await update.message.reply_text(
            "👋 Halo lagi! Gue masih inget percakapan kita.\n"
            "/reset untuk mulai fresh"
        )
    else:
        await update.message.reply_text(
            "👋 Halo! Gue *Adit Agent*.\n\n"
            "Bisa:\n"
            "🔍 Search info\n"
            "📝 Baca/tulis file\n"
            "🧠 Ingat chat history\n\n"
            "/help untuk info lengkap",
            parse_mode="Markdown"
        )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text("🧹 History dihapus!")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Adit Agent*\n\n"
        "/start  — Status\n"
        "/reset  — Clear history\n"
        "/help   — Info ini\n"
        "/ping   — Test\n\n"
        "_Cukup chat normal, gue handle sendiri_",
        parse_mode="Markdown"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏓 Pong")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🤖 Status\n\nDB: {DB_PATH}\n"
        f"History: {MAX_HISTORY}\n"
        f"Rate Limit: {RATE_LIMIT_SECONDS}s"
    )

# ======================
# MAIN
# ======================

def main():
    init_db()
    # test_llm()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    logger.info("🤖 Adit Agent Running (Optimized)...")
    app.run_polling()

if __name__ == "__main__":
    main()