"""
recognize_attendance.py
------------------------
Runs the live webcam recognition loop:
  1. Detects faces in each frame with a Haar cascade.
  2. Recognizes each face with the trained LBPH model.
  3. If the match confidence is good enough, marks attendance for that
     employee in attendance/<YYYY-MM-DD>.csv (see utils.mark_attendance).

Usage:
    python src/recognize_attendance.py

Press 'q' to quit the live window.
"""

import sys

import cv2

from config import CONFIDENCE_THRESHOLD, HAAR_CASCADE_PATH, TRAINER_MODEL_PATH
from utils import load_labels, mark_attendance


def run_recognition():
    labels = load_labels()
    if not labels:
        raise RuntimeError("No trained labels found. Run train_model.py after enrolling employees.")

    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_MODEL_PATH)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not access the webcam (device index 0).")

    print("[INFO] Starting recognition loop. Press 'q' to quit.")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            emp_id, distance = recognizer.predict(face_crop)

            # LBPH returns a distance: LOWER means a more confident match.
            if distance < CONFIDENCE_THRESHOLD and emp_id in labels:
                name = labels[emp_id]
                logged = mark_attendance(name)
                label_text = f"{name} ({distance:.0f})"
                color = (0, 255, 0)
                if logged:
                    print(f"[ATTENDANCE] {name} marked present.")
            else:
                label_text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_recognition()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
