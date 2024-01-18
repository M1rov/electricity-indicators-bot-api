from models import Report


class ReportService:
    def __init__(self, session):
        self.session = session

    def add_report(self, chat_id, message):
        new_report = Report(chat_id=chat_id, message=message)
        self.session.add(new_report)
        self.session.commit()
        return new_report
