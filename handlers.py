import telebot
from db import session
from get_response import get_response
from services.report import ReportService


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
            print(message.from_user.id, 'user id ')
        except Exception as error:
            ReportService(session).add_report(message.chat.id, message.text)
            bot.send_message(message.chat.id, error.__str__())
