from models import Electricity


class ElectricityService:
    def __init__(self, session):
        self.session = session

    def add_recording(self, user_id, kWh, month, year):
        new_recording = Electricity(user_id=user_id, month=month, year=year, kWh=kWh)
        self.session.add(new_recording)
        self.session.commit()
        return new_recording
