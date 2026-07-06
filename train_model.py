"""
train_model.py
----------------
Trains an OpenCV LBPH (Local Binary Patterns Histograms) face recognizer on
every employee folder found under dataset/, then saves:
    trainer/trainer.yml   -> the trained model
    trainer/labels.json   -> {numeric_id: employee_name} lookup table

Usage:
    python src/train_model.py
"""

import os
import sys

import cv2
import numpy as np

from config import DATASET_DIR, TRAINER_MODEL_PATH
from utils import save_labels


def gather_training_data():
    """
    Walk dataset/<emp_id>_<name>/*.jpg and build parallel lists of
    grayscale face images and their integer labels (emp_id).
    """
    faces, labels, label_names = [], [], {}

    if not os.path.isdir(DATASET_DIR):
        raise RuntimeError(f"Dataset directory not found: {DATASET_DIR}")

    person_folders = [d for d in os.listdir(DATASET_DIR)
                       if os.path.isdir(os.path.join(DATASET_DIR, d))]

    if not person_folders:
        raise RuntimeError(
            "No enrolled employees found. Run capture_faces.py first for each employee."
        )

    for folder in person_folders:
        try:
            emp_id_str, name = folder.split("_", 1)
            emp_id = int(emp_id_str)
        except ValueError:
            print(f"[WARN] Skipping folder with unexpected name format: {folder}")
            continue

        label_names[emp_id] = name.replace("_", " ")
        folder_path = os.path.join(DATASET_DIR, folder)

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(emp_id)

    return faces, labels, label_names


def train_and_save():
    faces, labels, label_names = gather_training_data()
    print(f"[INFO] Training on {len(faces)} images across {len(label_names)} employees ...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_MODEL_PATH)
    save_labels(label_names)

    print(f"[INFO] Model saved to {TRAINER_MODEL_PATH}")
    print(f"[INFO] Labels saved for: {list(label_names.values())}")


if __name__ == "__main__":
    try:
        train_and_save()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
