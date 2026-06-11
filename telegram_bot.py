import os
import json
import time
import sqlite3
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

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
def test_llm():
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "reply only OK"
                }
            ],
            max_tokens=5
        )

        logger.info("✅ LLM connection successful")

    except Exception as e:
        logger.error(f"❌ LLM connection failed: {e}")

MAX_HISTORY = 20        # jumlah pesan yang disimpan per user (10 pasang)
RATE_LIMIT_SECONDS = 5  # cooldown antar request per user

# Rate limiting (in-memory, tidak perlu persist)
user_last_request: dict[int, float] = {}

# ======================
# DATABASE — PERSISTENT MEMORY
# ======================
DB_PATH = os.getenv("DB_PATH", "memory.db")

def init_db():
    """Buat tabel kalau belum ada."""
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
    conn.commit()
    conn.close()
    logger.info(f"✅ Database initialized at {DB_PATH}")

def load_history(chat_id: int) -> list:
    """Load history terakhir untuk chat_id tertentu."""
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
    # Kembalikan dalam urutan kronologis (ASC)
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

def save_message(chat_id: int, role: str, content: str):
    """Simpan satu pesan ke database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_history (chat_id, role, content)
        VALUES (?, ?, ?)
    """, (chat_id, role, content))
    conn.commit()
    conn.close()

def clear_history(chat_id: int):
    """Hapus semua history untuk chat_id tertentu."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()

def trim_history(chat_id: int):
    """Hapus pesan lama kalau sudah melebihi MAX_HISTORY."""
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
    """Search menggunakan Tavily API — didesain khusus untuk AI agent."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query":   query,
            "max_results": 5,
            "search_depth": "basic",
            "include_answer": True   # Tavily bisa langsung kasih summary singkat
        }
        res = requests.post(url, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()

        output = []

        # Kalau Tavily kasih direct answer, tampilkan dulu
        if data.get("answer"):
            output.append(f"💡 *Ringkasan:* {data['answer']}")

        # Lalu tambahkan hasil individual
        for item in data.get("results", []):
            title   = item.get("title", "")
            url_    = item.get("url", "")
            content = item.get("content", "")
            output.append(f"📌 {title}\n   {content[:200]}...\n   🔗 {url_}")

        return "\n\n".join(output) if output else "No results found."

    except requests.exceptions.HTTPError as e:
        logger.error(f"Tavily search HTTP error: {e}")
        if e.response.status_code == 401:
            return "⚠️ Tavily API key tidak valid."
        if e.response.status_code == 429:
            return "⚠️ Tavily quota habis untuk hari ini."
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
# PLANNER
# ======================

def planner(user_input: str) -> dict:
    prompt = f"""
You are an AI agent planner.

User request:
{user_input}

Return STRICT JSON only:
{{
  "steps": [
    {{
      "tool": "web_search | write_file | read_file | none",
      "input": "string or object"
    }}
  ]
}}

Only return valid JSON. No explanation, no markdown.
"""

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        raw = res.choices[0].message.content.strip()

        logger.info(f"Planner output: {raw}")

        raw = (
            raw
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"Planner JSON parse error: {e}")
        return {"steps": []}

    except Exception as e:
        logger.error(f"Planner error: {e}")
        return {"steps": []}

# ======================
# TOOL EXECUTOR
# ======================

def execute_tool(step: dict):
    logger.info(f"Executing tool: {step}")

    tool = step.get("tool")
    inp  = step.get("input")

    if tool == "web_search":
        return web_search(inp)
    if tool == "write_file":
        if isinstance(inp, dict):
            return write_file(inp.get("path", "output.txt"), inp.get("content", ""))
        return "❌ write_file requires {path, content}"
    if tool == "read_file":
        return read_file(inp)
    return None

# ======================
# AGENT CORE
# ======================

def run_agent(user_input: str, chat_id: int) -> str:
    # Jalankan planner dan tools
    plan = planner(user_input)
    observations = []

    for step in plan.get("steps", []):
        result = execute_tool(step)
        if result:
            observations.append(result)

    # Load history dari SQLite (persistent!)
    history = load_history(chat_id)

    system_msg = {
        "role": "system",
        "content": (
            "You are Adit Agent, an advanced and helpful AI assistant. "
            "You remember the conversation history with this user. "
            "Answer clearly, helpfully, and in the same language the user uses. "
            "If tool results are provided, use them to enrich your answer."
        )
    }

    # Gabungkan tool results ke user message
    tool_context = ""

    if observations:
        tool_context = "\n\n[Tool Results]\n" + "\n\n".join(observations)

    current_user_msg = {
        "role": "user",
        "content": user_input + tool_context
    }

    messages = [system_msg] + history + [current_user_msg]

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )

        answer = res.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)

        return (
            "⚠️ AI Provider Error\n\n"
            f"{str(e)}"
        )

    # Simpan ke SQLite (persistent memory)
    save_message(chat_id, "user", user_input)
    save_message(chat_id, "assistant", answer)
    trim_history(chat_id)

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
    chat_id   = update.effective_chat.id
    user_text = update.message.text

    # Rate limiting
    now  = time.time()
    last = user_last_request.get(chat_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        await update.message.reply_text(
            f"⏳ Pelan-pelan ya! Tunggu {RATE_LIMIT_SECONDS} detik sebelum kirim lagi."
        )
        return
    user_last_request[chat_id] = now

    # Typing indicator
    await context.bot.send_chat_action(
    chat_id=chat_id,
    action=ChatAction.TYPING
)

    try:
        response = run_agent(user_text, chat_id)
        await send_long_message(update, response)
    except Exception as e:
        logger.error(f"handle error for chat_id {chat_id}: {e}")
        await update.message.reply_text(
            "❌ Maaf, terjadi error saat memproses permintaanmu. Coba lagi ya!"
        )

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    history = load_history(chat_id)
    if history:
        await update.message.reply_text(
            "👋 Halo lagi! Gue masih inget percakapan kita sebelumnya.\n"
            "Ketik /reset kalau mau mulai dari awal."
        )
    else:
        await update.message.reply_text(
            "👋 Halo! Gue *Adit Agent*, AI agent yang siap bantu kamu.\n\n"
            "Gue bisa:\n"
            "🔍 Search Google untuk info terkini\n"
            "📝 Baca & tulis file\n"
            "🧠 Ingat percakapan kita, bahkan setelah restart!\n\n"
            "Ketik /help untuk info lebih lanjut.",
            parse_mode="Markdown"
        )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    clear_history(chat_id)
    await update.message.reply_text(
        "🧹 History percakapan kamu sudah dihapus! Mulai fresh sekarang."
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Adit Agent — Help*\n\n"
        "*Commands:*\n"
        "/start  — Sapa bot & cek status memory\n"
        "/reset  — Hapus history percakapan\n"
        "/help   — Tampilkan pesan ini\n\n"
        "*Kemampuan:*\n"
        "• 🔍 Search Google (info terkini)\n"
        "• 📝 Baca & tulis file\n"
        "• 🧠 Ingat konteks percakapan secara permanen\n"
        "• 🌐 Jawab dalam bahasa yang kamu pakai\n\n"
        "_Cukup ketik pertanyaan atau perintahmu, agent akan otomatis menentukan tool yang tepat._",
        parse_mode="Markdown"
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"""
🤖 Adit Agent Status

DB_PATH: {DB_PATH}
MAX_HISTORY: {MAX_HISTORY}
RATE_LIMIT: {RATE_LIMIT_SECONDS}s
"""
    )
async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏓 Pong")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"""
🤖 Adit Agent Status

DB_PATH: {DB_PATH}
MAX_HISTORY: {MAX_HISTORY}
RATE_LIMIT: {RATE_LIMIT_SECONDS}s
"""
    )
# ======================
# MAIN
# ======================

def main():
    init_db()
    test_llm()  # Inisialisasi database saat startup

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    app   = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    logger.info("🤖 Adit Agent Running...")
    app.run_polling()

if __name__ == "__main__":
    main()