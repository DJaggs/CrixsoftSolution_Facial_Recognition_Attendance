"""
config.py
---------
Central configuration for the Facial Recognition Employee Attendance System.
Keeping paths and tunable parameters in one place makes the rest of the
codebase easier to read and to adapt to a real deployment.
"""

import os

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # captured face images
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")           # trained LBPH model + labels
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")     # daily attendance CSVs

TRAINER_MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_PATH = os.path.join(TRAINER_DIR, "labels.json")
EMPLOYEES_CSV = os.path.join(BASE_DIR, "employees.csv")

for directory in (DATASET_DIR, TRAINER_DIR, ATTENDANCE_DIR):
    os.makedirs(directory, exist_ok=True)

# ---------------------------------------------------------------------------
# Face detection / recognition parameters
# ---------------------------------------------------------------------------
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(__import__("cv2").__file__), "data", "haarcascade_frontalface_default.xml"
)

FACE_SIZE = (200, 200)          # every cropped face is resized to this before storing/training
SAMPLES_PER_EMPLOYEE = 60       # number of face images captured per employee during enrollment
CONFIDENCE_THRESHOLD = 65.0     # LBPH distance: LOWER is a better match. Above this -> "Unknown"

# Minimum minutes between two consecutive check-ins for the same employee on the
# same day. Prevents the camera loop from writing 30 rows a second while a
# person stands in front of it.
MIN_MINUTES_BETWEEN_LOGS = 1
