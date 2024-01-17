import telebot

from get_response import get_response


def setup_handlers(bot: telebot.TeleBot):
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Привіт! Я твій діалоговий агент для передачі показників електроенергії.")

    # Тут можна додати інші хендлери


    # Функція для обробки текстових повідомлень
    @bot.message_handler(func=lambda message: True)
    def handle_message(message):
        try:
            response = get_response(message.text)
            bot.send_message(message.chat.id, response)
        except Exception as error:
            bot.send_message(message.chat.id, error.__str__())
