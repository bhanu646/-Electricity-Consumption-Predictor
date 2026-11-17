"""
SQLite Database module for recording and retrieving electricity consumption predictions.
Keeps historical records for review, reporting, and auditing.
"""

import os
import sqlite3
from datetime import datetime
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "predictions.db")

def init_db(db_path=DB_PATH):
    """Initializes the SQLite database and creates the predictions table if it does not exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            number_of_people INTEGER NOT NULL,
            previous_day_consumption REAL NOT NULL,
            day_of_week TEXT NOT NULL,
            is_weekend INTEGER NOT NULL,
            appliance_usage REAL NOT NULL,
            predicted_consumption REAL NOT NULL,
            model_used TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(temperature, humidity, number_of_people, previous_day_consumption,
                    day_of_week, is_weekend, appliance_usage, predicted_consumption, model_used,
                    db_path=DB_PATH):
    """Saves a single prediction record into the SQLite database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO prediction_history (
            timestamp, temperature, humidity, number_of_people,
            previous_day_consumption, day_of_week, is_weekend,
            appliance_usage, predicted_consumption, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp_str,
        float(temperature),
        float(humidity),
        int(number_of_people),
        float(previous_day_consumption),
        str(day_of_week),
        int(is_weekend),
        float(appliance_usage),
        float(predicted_consumption),
        str(model_used)
    ))
    conn.commit()
    conn.close()

def get_all_predictions(db_path=DB_PATH, limit=100):
    """Retrieves prediction history as a pandas DataFrame."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM prediction_history ORDER BY id DESC LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def clear_predictions(db_path=DB_PATH):
    """Clears all records in the predictions table."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
