import sqlalchemy

from models.base import Base


class Electricity(Base):
    __tablename__ = "electricity"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    year = sqlalchemy.Column(sqlalchemy.Integer,sqlalchemy.CheckConstraint("year > 2000 AND year <= date_part('year', CURRENT_DATE)"), nullable=False)
    month = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.CheckConstraint('month >= 0 AND month < 12'),  nullable=False)
    kWh = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)

