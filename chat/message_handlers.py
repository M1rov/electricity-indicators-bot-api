from datetime import date

from telebot import types

from chat.utils import get_months_plural, format_electricity_recording
from consts import months, monthsGenitive
from services.electricity import ElectricityService


class MessageHandler:
    def __init__(self, bot):
        self.bot = bot

    def ask_yes_or_no(self, message, question, next_step):
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add('Так', 'Ні')
        self.bot.send_message(message.chat.id, question, reply_markup=markup)
        self.bot.register_next_step_handler(message, lambda m: self.process_yes_or_no_response(m, next_step))

    def process_yes_or_no_response(self, message, next_step):
        if "так" in message.text.lower():
            next_step(message)
        elif "ні" in message.text.lower():
            self.bot.send_message(message.chat.id, "Зрозумів, тоді може я можу ще чимось Вам допомогти?")
        else:
            self.ask_yes_or_no(message, 'Дайте відповідь "так" чи "ні", будь ласка', next_step)


class AddRecordingHandler(MessageHandler):
    def __init__(self, bot):
        super().__init__(bot)
        self.month = None
        self.year = None

    def start(self, msg):
        self.step_one(msg)

    def step_one(self, message):
        self.ask_yes_or_no(message, 'Ви хочете передати показники електроенергії?', self.step_two)

    def step_two(self, message):
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        current_year = date.today().year
        years_list = [str(current_year - 2), str(current_year - 1), str(current_year)]
        markup.add(*years_list)
        msg = self.bot.send_message(message.chat.id, 'Для початку, введіть рік, за який Ви передаєте показники', reply_markup=markup)
        self.bot.register_next_step_handler(msg, self.step_three)

    def step_three(self, message):
        self.year = message.text
        today = date.today()
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        if int(self.year) == today.year:
            markup.add(*months[0:today.month])
        else:
            markup.add(*months)
        msg = self.bot.send_message(message.chat.id, 'Добре, за який місяць Ви хочете передати показники?', reply_markup=markup)
        self.bot.register_next_step_handler(msg, self.step_four)

    def step_four(self, message):
        self.month = months.index(message.text)
        msg = self.bot.send_message(message.chat.id, 'Ми майже на фініші :)\nНапишіть будь ласка, показники електроенергії з лічильника')
        self.bot.register_next_step_handler(msg, self.step_five)

    def step_five(self, message):
        kWh = message.text
        ElectricityService().add_recording(message.from_user.id, kWh, self.month, self.year)
        self.bot.send_message(message.chat.id, 'Дані успішно записані! Чи є ще щось, чим я можу вам допомогти?')


class ViewRecordingHandler(MessageHandler):
    def __init__(self, bot):
        super().__init__(bot)
        self.month = None
        self.year = None
        self.period = None

    def start(self, msg):
        self.step_one(msg)

    def step_one(self, message):
        self.ask_yes_or_no(message, 'Ви хочете переглянути показники електроенергії?', self.step_two)

    def step_two(self, message):
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add('Останній місяць', 'Останній рік', 'Інший період')
        msg = self.bot.send_message(message.chat.id, 'За який період Ви хочете переглянути показники?', reply_markup=markup)
        self.bot.register_next_step_handler(msg, self.step_three)

    def step_three(self, message):
        period = message.text
        if period == 'Останній місяць':
            self.period = 1
            self.month = date.today().month - 1
            self.year = date.today().year
            self.send_statistics(message)
        elif period == 'Останній рік':
            self.period = 12
            self.month = date.today().month
            self.year = date.today().year - 1
            self.send_statistics(message)
        else:
            markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
            markup.add(*[str(i + 1) for i in range(12)])
            msg = self.bot.send_message(message.chat.id, 'Оберіть кількість місяців для перегляду показників', reply_markup=markup)
            self.bot.register_next_step_handler(msg, self.step_four)

    def step_four(self, message):
        self.period = int(message.text)
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        markup.add(*months)
        msg = self.bot.send_message(message.chat.id, 'Починаючи з якого місяця?', reply_markup=markup)
        self.bot.register_next_step_handler(msg, self.step_five)

    def step_five(self, message):
        self.month = months.index(message.text)
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        current_year = date.today().year
        years_list = [str(current_year - 2), str(current_year - 1), str(current_year)]
        markup.add(*years_list)
        msg = self.bot.send_message(message.chat.id, 'Поточний рік, чи інший?', reply_markup=markup)
        self.bot.register_next_step_handler(msg, self.step_six)

    def step_six(self, message):
        self.year = int(message.text)
        self.send_statistics(message)

    def send_statistics(self, message):
        self.bot.send_message(message.chat.id,
                              f'Виводжу інформацію за {f"{self.period} {get_months_plural(self.period)} (починаючи з {monthsGenitive[self.month]} {self.year} року)" if self.period > 1 else f"{months[self.month].lower()} {self.year} року"}')
        recordings = ElectricityService().get_recording(message.from_user.id, self.period, self.month, self.year)
        print(recordings, 'test')
        self.bot.send_message(message.chat.id, '\n'.join([format_electricity_recording(rec) for rec in recordings]))



message_handlers = {
    "add_recording": AddRecordingHandler,
    "view_recording": ViewRecordingHandler
}
