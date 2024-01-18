import sqlalchemy

from models.base import Base


class Report(Base):
    __tablename__ = "reports"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    chat_id = sqlalchemy.Column(sqlalchemy.Text, nullable=False)
    message = sqlalchemy.Column(sqlalchemy.Text, nullable=False)

