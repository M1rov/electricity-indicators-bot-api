from datetime import datetime

from db import session
from models import Electricity


class ElectricityService:
    def __init__(self):
        self.session = session

    def add_recording(self, user_id, kWh, month, year):
        new_recording = Electricity(user_id=user_id, month=month, year=year, kWh=kWh)
        self.session.add(new_recording)
        self.session.commit()
        return new_recording

    def get_recording(self, user_id, period, start_month, start_year):
        # Розрахунок дати завершення на основі періоду
        end_year = start_year + ((start_month + period - 1) // 12)
        end_month = (start_month + period - 1) % 12 + 1

        print(user_id, period, start_month, start_year, end_month, end_year, 'debug!')

        recordings = self.session.query(Electricity) \
            .filter(Electricity.user_id == user_id) \
            .filter(
            (Electricity.year > start_year) |
            ((Electricity.year == start_year) & (Electricity.month >= start_month))) \
            .filter(
            (Electricity.year < end_year) |
            ((Electricity.year == end_year) & (Electricity.month < end_month))) \
            .all()

        return recordings
