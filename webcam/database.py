import os
import pickle
from config import DATABASE_FILE

def load_database():
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(db, f)
