"""
capture_faces.py
-----------------
Enrollment script: registers a new employee by capturing face samples from
the webcam.

Usage:
    python src/capture_faces.py --name "John Doe" --emp-id 101

Each employee gets a numeric emp-id (used internally by the LBPH recognizer)
and their captured face crops are stored under:
    dataset/<emp_id>_<name>/*.jpg
"""

import argparse
import os
import sys

import cv2

from config import DATASET_DIR, FACE_SIZE, HAAR_CASCADE_PATH, SAMPLES_PER_EMPLOYEE


def capture_employee_faces(emp_id: int, name: str, samples: int = SAMPLES_PER_EMPLOYEE) -> str:
    detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {HAAR_CASCADE_PATH}")

    safe_name = name.strip().replace(" ", "_")
    person_dir = os.path.join(DATASET_DIR, f"{emp_id}_{safe_name}")
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Could not access the webcam (device index 0).")

    print(f"[INFO] Look at the camera. Capturing {samples} samples for '{name}' (ID {emp_id}).")
    count = 0

    while count < samples:
        ok, frame = cam.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face_crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_crop)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}/{samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break  # only take one face per frame to avoid capturing bystanders

        cv2.imshow("Enrollment - press Q to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} face samples into {person_dir}")
    return person_dir


def main():
    parser = argparse.ArgumentParser(description="Enroll a new employee's face for attendance.")
    parser.add_argument("--name", required=True, help="Employee full name, e.g. 'John Doe'")
    parser.add_argument("--emp-id", required=True, type=int, help="Unique numeric employee ID")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_EMPLOYEE,
                         help="Number of face images to capture (default: %(default)s)")
    args = parser.parse_args()

    try:
        capture_employee_faces(args.emp_id, args.name, args.samples)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
