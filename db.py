from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DATABASE_URL
from models import Base

engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)
session = Session()

# Створення таблиць у БД
Base.metadata.create_all(engine)
