import telebot
from config import BOT_TOKEN
from handlers import setup_handlers


bot = telebot.TeleBot(BOT_TOKEN)

# Налаштування хендлерів
setup_handlers(bot)

# Постійний polling
bot.polling()
