import os
import random
from pyrogram import Client, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

# Translation
from deep_translator import GoogleTranslator
from langdetect import detect

load_dotenv()

# ----- ENV VARIABLES -----
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

# ----- BOT CLIENT -----
app = Client("ai_gf_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# ----- LOAD MODEL -----
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ----- TRANSLATOR WRAPPER -----
class TranslatorWrapper:
    def translate(self, text, dest='en'):
        return GoogleTranslator(source='auto', target=dest).translate(text)
    
    def detect(self, text):
        return type('obj', (), {'lang': detect(text)})

translator = TranslatorWrapper()
chat_history = {}

# ----- MOODS -----
moods = {
    "happy": ["happy", "lol", "haha", "yay", "funny", "good"],
    "romantic": ["love", "miss", "heart", "romantic", "darling", "sweet"],
    "sad": ["sad", "lonely", "cry", "pain", "depressed"],
    "angry": ["angry", "mad", "hate", "frustrated", "annoyed"]
}

mood_media = {
    "happy": ["CAACAgIAAxkBAAEGhWFiw5jL3V6O1QyJ1JXSljG5GGL6VwACpQADVp29Cl2t3GVmTnZlIwQ"],
    "romantic": ["CAACAgIAAxkBAAEGhWFiw5iUgfP4rJ0gQ7Hx7VjB9iAAfZQACpQADVp29Ck4i9ZfIEXj7IwQ"],
    "sad": ["CAACAgIAAxkBAAEGhWFiw5jZC7dqE-Jb8vWgHtF4Jv0ppwACpwADVp29CnY1vhLwY3MUwQ"],
    "angry": ["CAACAgIAAxkBAAEGhWFiw5jao2U1xZqEcxzHTm6CnD_V_gACqgADVp29Cq1Z84fpfrK4IwQ"]
}

mood_emojis = {
    "happy": "😄🎉✨",
    "romantic": "❤️💌😍",
    "sad": "😢💔😔",
    "angry": "😠🔥😡"
}

# ----- LOAD QUOTES -----
def load_quotes(mood):
    path = f"quotes/{mood}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []

# ----- DETECT MOOD -----
def detect_mood(text):
    text = text.lower()
    for mood, keywords in moods.items():
        if any(word in text for word in keywords):
            return mood
    return None

# ----- BOT MESSAGE HANDLER -----
@app.on_message(filters.private | filters.group)
async def ai_gf_reply(client, message):
    if not message.text:
        return

    user_id = message.from_user.id
    username = message.from_user.first_name or "User"
    text = message.text

    # Translate to English for model
    translated_text = translator.translate(text, dest='en')

    # Chat history
    if user_id in chat_history:
        previous = chat_history[user_id]
    else:
        previous = tokenizer.encode("", return_tensors="pt")

    new_input = tokenizer.encode(translated_text + tokenizer.eos_token, return_tensors="pt")
    bot_input = torch.cat([previous, new_input], dim=-1)

    chat_history_ids = model.generate(bot_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply_en = tokenizer.decode(chat_history_ids[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    chat_history[user_id] = chat_history_ids

    # Translate reply back to user language
    reply = translator.translate(reply_en, dest=translator.detect(text).lang)

    # Mood detection
    mood = detect_mood(text)
    if mood:
        media_list = mood_media.get(mood, [])
        if media_list:
            media_id = random.choice(media_list)
            await message.reply_sticker(media_id)

        quotes = load_quotes(mood)
        if quotes:
            reply += f"\n\n💌 {random.choice(quotes)}"

        reply += f" {mood_emojis.get(mood,'')}"

    # Add personalized tag in groups
    if message.chat.type != "private":
        reply = f"@{message.from_user.username or username}, {reply}"

    # Final signature
    reply += "\n\n— my Boyfriend @Lippsxd ❤️"
    reply += "\nMade by @Lippsxd"

    await message.reply_text(reply)

print("Ultimate AI Girlfriend/Boyfriend Bot is running...")
app.run()
