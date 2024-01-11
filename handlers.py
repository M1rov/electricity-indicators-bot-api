import telebot


def setup_handlers(bot: telebot.TeleBot):
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Привіт! Я твій діалоговий агент для передачі показників електроенергії.")

    # Тут можна додати інші хендлери
