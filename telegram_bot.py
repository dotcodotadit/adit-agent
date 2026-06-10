import os
import json
import logging
import requests
from collections import defaultdict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# ======================
# LOAD ENV
# ======================
load_dotenv(dotenv_path=".env")

# ======================
# LOGGING (ganti print debug)
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

# ======================
# CONFIG
# ======================
client = OpenAI(
    api_key=os.getenv("FREEMODEL_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.freemodel.dev/v1")
)

# Conversation history per chat_id
conversation_history: dict[int, list] = defaultdict(list)
MAX_HISTORY = 10  # simpan max 10 pesan terakhir per user

# Rate limiting: simpan timestamp terakhir per user
import time
user_last_request: dict[int, float] = {}
RATE_LIMIT_SECONDS = 5  # cooldown antar request

# ======================
# TOOLS
# ======================

def web_search(query: str) -> str:
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select(".result__a")[:5]:
            title = a.get_text()
            link = a.get("href")
            results.append(f"{title} | {link}")
        return "\n".join(results) if results else "No results found"
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
        logger.error(f"write_file error: {e}")
        return f"❌ Failed to write file: {str(e)}"

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

Only return valid JSON. No explanation.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = res.choices[0].message.content.strip()
        # Strip markdown code fences kalau ada
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
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

def execute_tool(step: dict) -> str | None:
    tool = step.get("tool")
    inp = step.get("input")

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
    plan = planner(user_input)
    observations = []

    for step in plan.get("steps", []):
        result = execute_tool(step)
        if result:
            observations.append(result)

    # Bangun messages dengan history percakapan
    history = conversation_history[chat_id]

    system_msg = {
        "role": "system",
        "content": (
            "You are an advanced AI agent. "
            "Answer clearly and helpfully. "
            "If tool results are provided, use them to enrich your answer."
        )
    }

    context_msg = {
        "role": "user",
        "content": f"""User request: {user_input}

Plan executed:
{json.dumps(plan, indent=2)}

Tool results:
{chr(10).join(observations) if observations else "No tools used."}

Give a clear, helpful final answer."""
    }

    messages = [system_msg] + history + [context_msg]

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )

    answer = res.choices[0].message.content

    # Update conversation history
    conversation_history[chat_id].append({"role": "user", "content": user_input})
    conversation_history[chat_id].append({"role": "assistant", "content": answer})

    # Trim history kalau sudah terlalu panjang
    if len(conversation_history[chat_id]) > MAX_HISTORY * 2:
        conversation_history[chat_id] = conversation_history[chat_id][-(MAX_HISTORY * 2):]

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
# TELEGRAM HANDLER
# ======================

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Rate limiting
    now = time.time()
    last = user_last_request.get(chat_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        await update.message.reply_text(
            f"⏳ Pelan-pelan ya! Tunggu {RATE_LIMIT_SECONDS} detik sebelum kirim lagi."
        )
        return
    user_last_request[chat_id] = now

    # Typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        response = run_agent(user_text, chat_id)
        await send_long_message(update, response)
    except Exception as e:
        logger.error(f"handle error for chat_id {chat_id}: {e}")
        await update.message.reply_text(
            "❌ Maaf, terjadi error saat memproses permintaanmu. Coba lagi ya!"
        )

# ======================
# COMMAND: /reset — hapus history chat
# ======================

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversation_history[chat_id].clear()
    await update.message.reply_text("🧹 History percakapan kamu sudah dihapus!")

# ======================
# MAIN
# ======================

def main():
    from telegram.ext import CommandHandler

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    logger.info("🤖 Autonomous Agent Running...")
    app.run_polling()

if __name__ == "__main__":
    main()