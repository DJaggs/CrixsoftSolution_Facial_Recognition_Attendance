"""
utils.py
--------
Small shared helpers used by the enrollment, training, and recognition
scripts: label bookkeeping (name <-> numeric id) and attendance CSV writing.
"""

import csv
import json
import os
from datetime import datetime

from config import ATTENDANCE_DIR, LABELS_PATH, MIN_MINUTES_BETWEEN_LOGS


def load_labels() -> dict:
    """Return {numeric_id (int): employee_name (str)}. Empty dict if not trained yet."""
    if not os.path.exists(LABELS_PATH):
        return {}
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_labels(labels: dict) -> None:
    """Persist {numeric_id: employee_name} to disk as JSON."""
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in labels.items()}, f, indent=2)


def _today_csv_path() -> str:
    filename = datetime.now().strftime("%Y-%m-%d") + ".csv"
    return os.path.join(ATTENDANCE_DIR, filename)


def _last_logged_time(csv_path: str, employee_name: str):
    """Return the datetime of the employee's last log today, or None."""
    if not os.path.exists(csv_path):
        return None
    last_time = None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Name"] == employee_name:
                last_time = datetime.strptime(row["Time"], "%H:%M:%S")
    return last_time


def mark_attendance(employee_name: str) -> bool:
    """
    Append a (Name, Date, Time) row to today's attendance CSV for the given
    employee, unless they were already logged within MIN_MINUTES_BETWEEN_LOGS.

    Returns True if a new row was written, False if it was skipped as a
    duplicate/too-soon entry.
    """
    csv_path = _today_csv_path()
    now = datetime.now()

    last_time = _last_logged_time(csv_path, employee_name)
    if last_time is not None:
        elapsed_minutes = (now - now.replace(
            hour=last_time.hour, minute=last_time.minute, second=last_time.second
        )).total_seconds() / 60.0
        if abs(elapsed_minutes) < MIN_MINUTES_BETWEEN_LOGS:
            return False

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Date", "Time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "Name": employee_name,
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S"),
        })
    return True
