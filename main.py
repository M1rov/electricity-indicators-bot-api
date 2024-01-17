import telebot
from config import BOT_TOKEN
from handlers import setup_handlers


bot = telebot.TeleBot(BOT_TOKEN)

# Налаштування хендлерів
setup_handlers(bot)

print('Bot started!')

# Постійний polling
bot.polling()
