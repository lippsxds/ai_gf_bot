import os
import random
from pyrogram import Client, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("BOT_TOKEN")

app = Client("ai_gf_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Lightweight model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history = {}

# Mood keywords
moods = {
    "happy": ["happy","lol","haha","yay","funny","good"],
    "romantic": ["love","miss","heart","romantic","darling","sweet"],
    "sad": ["sad","lonely","cry","pain","depressed"],
    "angry": ["angry","mad","hate","frustrated","annoyed"]
}

# Mood emojis
mood_emojis = {
    "happy": "😄🎉✨",
    "romantic": "❤️💌😍",
    "sad": "😢💔😔",
    "angry": "😠🔥😡"
}

# Stickers for moods
mood_stickers = {
    "happy": ["CAACAgIAAxkBAAEGhWFiw5jL3V6O1QyJ1JXSljG5GGL6VwACpQADVp29Cl2t3GVmTnZlIwQ"],
    "romantic": ["CAACAgIAAxkBAAEGhWFiw5iUgfP4rJ0gQ7Hx7VjB9iAAfZQACpQADVp29Ck4i9ZfIEXj7IwQ"],
    "sad": ["CAACAgIAAxkBAAEGhWFiw5jZC7dqE-Jb8vWgHtF4Jv0ppwACpwADVp29CnY1vhLwY3MUwQ"],
    "angry": ["CAACAgIAAxkBAAEGhWFiw5jao2U1xZqEcxzHTm6CnD_V_gACqgADVp29Cq1Z84fpfrK4IwQ"]
}

# Load quotes
def load_quotes(mood):
    path = f"quotes/{mood}.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []

# Detect mood
def detect_mood(text):
    text = text.lower()
    for mood, keywords in moods.items():
        if any(word in text for word in keywords):
            return mood
    return None

@app.on_message(filters.private | filters.group)
async def ai_reply(client, message):
    if not message.text:
        return

    text = message.text
    user_id = message.from_user.id
    username = message.from_user.first_name or "User"

    # Chat history (last 2 messages)
    prev = chat_history.get(user_id, [])
    new_input = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input = torch.cat(prev + [new_input], dim=-1) if prev else new_input

    chat_ids = model.generate(bot_input, max_length=100, pad_token_id=tokenizer.eos_token_id)
    reply_text = tokenizer.decode(chat_ids[:, bot_input.shape[-1]:][0], skip_special_tokens=True)

    chat_history[user_id] = (prev + [new_input])[-2:]

    # Mood detection
    mood = detect_mood(text)
    if mood:
        # Send sticker
        stickers = mood_stickers.get(mood, [])
        if stickers:
            await message.reply_sticker(random.choice(stickers))

        # Append quote
        quotes = load_quotes(mood)
        if quotes:
            reply_text += f"\n\n💌 {random.choice(quotes)}"

        reply_text += f" {mood_emojis.get(mood,'')}"

    # Mention in group
    if message.chat.type != "private":
        reply_text = f"@{message.from_user.username or username}, {reply_text}"

    # Signature
    reply_text += "\n\n— my boyFriend @lippsxd ❤️"

    await message.reply_text(reply_text)

print("Lightweight AI Telegram Bot with group & stickers is running...")
app.run()