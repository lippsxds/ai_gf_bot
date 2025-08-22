import os
import asyncio
import random
import re

from pyrogram import Client, filters
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import httpx

# ----- Load Environment Variables -----
load_dotenv()
API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Hugging Face token

if not all([API_ID, API_HASH, BOT_TOKEN, HF_API_TOKEN]):
    raise ValueError("One or more environment variables are missing!")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
MODEL = "google/flan-t5-small"  # Ultra-light, fast model

# ----- Initialize Bot -----
bot = Client(
    "MiniChatGPTBot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ----- Helper Functions -----
def escape_markdown(text: str) -> str:
    """Escape special characters for Markdown v2."""
    return re.sub(r'([_*\[\]()~`>#+-=|{}.!])', r'\\\1', text)

# ----- AI Reply Function -----
async def get_ai_reply(text: str) -> str:
    """Send text to Hugging Face model and return AI response."""
    try:
        payload = {"inputs": text, "options": {"use_cache": True}}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{MODEL}",
                headers=HEADERS,
                json=payload
            )
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif "error" in data:
            return f"⚠️ Error: {data['error']}"
        else:
            return str(data)
    except httpx.TimeoutException:
        return "⚠️ Sorry, the AI took too long. Try again!"
    except Exception as e:
        return f"⚠️ Exception: {e}"

# ----- Handle Messages in Groups and DMs -----
@bot.on_message(filters.text & ~filters.bot)
async def handle_message(client, message):
    try:
        # Human-like delay
        await asyncio.sleep(random.uniform(0.2, 0.8))
        await message.chat.send_action("typing")

        # Only reply in groups if mentioned OR random chance
        if message.chat.type in ["group", "supergroup"]:
            bot_username = (await bot.get_me()).username.lower()
            if not message.text.lower().startswith(f"@{bot_username}") and random.randint(1, 10) > 1:
                return

        # Detect language
        user_lang = detect(message.text)

        # Translate to English if needed
        text_for_ai = GoogleTranslator(source=user_lang, target="en").translate(message.text) if user_lang != "en" else message.text

        # Get AI reply
        ai_reply_en = await get_ai_reply(text_for_ai)

        # Translate back if needed
        ai_reply = GoogleTranslator(source="en", target=user_lang).translate(ai_reply_en) if user_lang != "en" else ai_reply_en

        # Prepare mention safely
        mention = ""
        if message.from_user:
            mention = f"[{escape_markdown(message.from_user.first_name)}](tg://user?id={message.from_user.id})"

        # Send reply
        await message.reply_text(f"{mention}, {ai_reply}", parse_mode="markdown_v2")

    except Exception as e:
        await message.reply_text(f"⚠️ Exception: {e}")

# ----- Run Bot -----
bot.run()