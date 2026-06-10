import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

import json
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# DEBUG CHECK (hapus nanti kalau production)
print("TOKEN:", os.getenv("TELEGRAM_BOT_TOKEN"))
print("API:", os.getenv("FREEMODEL_API_KEY"))
print("BASE:", os.getenv("BASE_URL"))

# SAFETY CHECK
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise Exception("Missing TELEGRAM_BOT_TOKEN in .env")

# ======================
# CONFIG
# ======================

client = OpenAI(
    api_key=os.getenv("FREEMODEL_API_KEY"),
    base_url=os.getenv("BASE_URL", "https://api.freemodel.dev/v1")
)

# ======================
# TOOLS
# ======================

def web_search(query: str):
    url = f"https://duckduckgo.com/html/?q={query}"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

    soup = BeautifulSoup(res.text, "html.parser")

    results = []
    for a in soup.select(".result__a")[:5]:
        title = a.get_text()
        link = a.get("href")
        results.append(f"{title} | {link}")

    return "\n".join(results) if results else "No results found"


def write_file(path: str, content: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ File created: {path}"


def read_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ======================
# ROUTER (simple brain)
# ======================

def classify_task(text: str):
    text = text.lower()

    if any(w in text for w in ["code", "bug", "error", "script", "function", "fix"]):
        return "coding"

    if any(w in text for w in ["why", "explain", "analyze", "compare"]):
        return "reasoning"

    if "file" in text or "create" in text:
        return "file"

    return "fast"


# ======================
# PLANNER
# ======================

def planner(user_input: str):
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

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    try:
        return json.loads(res.choices[0].message.content)
    except:
        return {"steps": []}


# ======================
# TOOL EXECUTOR
# ======================

def execute_tool(step):
    tool = step.get("tool")
    inp = step.get("input")

    if tool == "web_search":
        return web_search(inp)

    if tool == "write_file":
        return write_file(inp["path"], inp["content"])

    if tool == "read_file":
        return read_file(inp)

    return None


# ======================
# AGENT CORE
# ======================

def run_agent(user_input: str):
    plan = planner(user_input)

    observations = []

    for step in plan.get("steps", []):
        result = execute_tool(step)
        if result:
            observations.append(result)

    final_prompt = f"""
You are an advanced AI agent.

User request:
{user_input}

Plan:
{json.dumps(plan, indent=2)}

Tool results:
{observations}

Give a clear, helpful final answer.
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.7
    )

    return res.choices[0].message.content


# ======================
# TELEGRAM HANDLER
# ======================

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = run_agent(user_text)
    await update.message.reply_text(response)


# ======================
# MAIN
# ======================

def main():
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🤖 Autonomous Agent Running...")
    app.run_polling()


if __name__ == "__main__":
    main()