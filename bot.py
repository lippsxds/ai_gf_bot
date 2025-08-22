import os
import requests
from pyrogram import Client, filters
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import asyncio
import random

# ----- Load Environment Variables -----
load_dotenv()
API_ID = int(os.environ.get("API_ID"))
API_HASH = os.environ.get("API_HASH")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # Hugging Face token

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
MODEL = "tiiuae/falcon-7b-instruct"  # Free Hugging Face model

# ----- Initialize Bot -----
bot = Client(
    "MiniChatGPTBot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# ----- AI Reply Function -----
async def get_ai_reply(text):
    try:
        payload = {"inputs": text, "options": {"use_cache": False}}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers=HEADERS,
            json=payload,
            timeout=60
        )
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif "error" in data:
            return f"⚠️ Error: {data['error']}"
        else:
            return str(data)
    except Exception as e:
        return f"⚠️ Exception: {e}"

# ----- Handle Messages in Groups and DMs -----
@bot.on_message(filters.text & ~filters.bot)
async def handle_message(client, message):
    try:
        await asyncio.sleep(random.uniform(0.5, 2.5))  # Human-like delay
        await message.chat.send_action("typing")

        # In groups, reply only when bot is mentioned OR random chance
        if message.chat.type in ["group", "supergroup"]:
            bot_username = (await bot.get_me()).username.lower()
            if bot_username not in message.text.lower() and random.randint(1, 5) > 1:
                return

        # Detect language
        user_lang = detect(message.text)

        # Translate to English if needed
        if user_lang != "en":
            text_for_ai = GoogleTranslator(source=user_lang, target="en").translate(message.text)
        else:
            text_for_ai = message.text

        # Get AI reply
        ai_reply_en = await get_ai_reply(text_for_ai)

        # Translate back to user's language if needed
        if user_lang != "en":
            ai_reply = GoogleTranslator(source="en", target=user_lang).translate(ai_reply_en)
        else:
            ai_reply = ai_reply_en

        # Prepare mention
        if message.from_user:
            mention = f"[{message.from_user.first_name}](tg://user?id={message.from_user.id})"
        else:
            mention = ""

        # Reply with mention
        await message.reply_text(f"{mention}, {ai_reply}", parse_mode="markdown")

    except Exception as e:
        await message.reply_text(f"⚠️ Exception: {e}")

# ----- Run Bot -----
bot.run()