# Facial Recognition Employee Attendance System

An AI-powered attendance system that uses **face detection + face recognition**
to automatically log employee check-ins — no manual entry, no swipe cards,
no biometric fingerprint hardware.

Built as part of the **Crixsoft Solution Artificial Intelligence Internship** (Project 1).

---

## Why this project

- **Automatic**: attendance is logged the moment a recognized employee steps in front of the camera.
- **Secure**: only faces the system has been trained on are marked present; unrecognized faces are flagged `Unknown` and are never logged.
- **Auditable**: every check-in is timestamped and saved to a plain CSV file per day, easy to import into payroll/HR tools.

---

## How it works

```
                ┌────────────────────┐
   Webcam  ───► │  Haar Cascade      │   detects face bounding boxes
                │  (face detection)  │
                └─────────┬──────────┘
                          │ cropped, grayscale face
                          ▼
                ┌────────────────────┐
                │  LBPH Recognizer   │   compares against trained model
                │ (face recognition) │
                └─────────┬──────────┘
                          │ employee_id + confidence score
                          ▼
                ┌────────────────────┐
                │  Attendance Logger │   writes Name, Date, Time → CSV
                └────────────────────┘
```

The system uses OpenCV's **Local Binary Patterns Histograms (LBPH)**
recognizer. It's lightweight, doesn't need a GPU, and works well for a
small/medium roster of enrolled employees — a solid, explainable choice for
a learning project (versus a deep-learning embedding model, which needs more
data and compute).

---

## Project structure

```
Facial-Recognition-Attendance-System/
├── src/
│   ├── config.py                # paths & tunable parameters
│   ├── utils.py                 # label + CSV attendance helpers
│   ├── capture_faces.py         # Step 1: enroll a new employee
│   ├── train_model.py           # Step 2: train the recognizer
│   └── recognize_attendance.py  # Step 3: run live attendance
├── dataset/                     # captured face images (per employee) — gitignored
├── trainer/                     # trained model + labels.json — gitignored
├── attendance/                  # daily attendance CSVs — gitignored
├── employees.csv                # sample employee roster
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> Face images, the trained model, and attendance logs are excluded from
> version control (see `.gitignore`) since they are personal biometric /
> runtime data, not source code. Each user enrolls and trains locally.

---

## Getting started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/Crixsoft-Solution_Facial-Recognition-Attendance-System.git
cd Crixsoft-Solution_Facial-Recognition-Attendance-System
pip install -r requirements.txt
```

Requires Python 3.9+ and a webcam.

### 2. Enroll each employee

Run once per employee, giving each one a unique numeric ID:

```bash
python src/capture_faces.py --name "John Doe" --emp-id 101
python src/capture_faces.py --name "Jane Smith" --emp-id 102
```

This opens the webcam and captures 60 face samples per person (adjustable
with `--samples`), storing them under `dataset/<emp_id>_<name>/`.

### 3. Train the recognizer

```bash
python src/train_model.py
```

This trains the LBPH model on every enrolled employee and saves
`trainer/trainer.yml` and `trainer/labels.json`.

### 4. Run live attendance

```bash
python src/recognize_attendance.py
```

A window opens showing the camera feed with a green box + name for
recognized employees (attendance is logged automatically, once per minute
minimum) and a red box + `Unknown` for anyone not enrolled. Press **q** to
quit.

Attendance is saved to `attendance/YYYY-MM-DD.csv`:

```
Name,Date,Time
John Doe,2026-07-06,09:02:14
Jane Smith,2026-07-06,09:05:41
```

---

## Configuration

Tunable values live in `src/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `SAMPLES_PER_EMPLOYEE` | 60 | face images captured per enrollment |
| `CONFIDENCE_THRESHOLD` | 65.0 | LBPH distance cutoff — lower distance = better match; above this the face is `Unknown` |
| `MIN_MINUTES_BETWEEN_LOGS` | 1 | prevents duplicate rows while a person lingers in frame |

Lower `CONFIDENCE_THRESHOLD` for stricter matching (fewer false positives,
more false "Unknown"s); raise it if legitimate employees are being
misclassified as `Unknown`.

---

## Limitations & possible improvements

This is an educational/portfolio implementation. Before any real HR/security
use, consider:

- **Liveness detection** — the current system can be fooled by a printed
  photo or a video played on a phone. A production system needs blink
  detection, depth sensing, or a challenge-response check.
- **Lighting sensitivity** — LBPH accuracy drops in poor/uneven lighting;
  a deep embedding model (e.g. FaceNet/ArcFace) is more robust but heavier.
- **Data privacy** — facial data is sensitive biometric information. A real
  deployment needs encryption at rest, consent tracking, and a retention/
  deletion policy compliant with local law.
- **Multi-face frames** — the enrollment script only captures one face per
  frame to avoid enrolling bystanders; the recognition script does handle
  multiple simultaneous faces.

---

## Tech stack

- Python 3
- OpenCV (`opencv-contrib-python`) — Haar Cascade detection + LBPH recognition
- NumPy, Pandas

---

## About this internship task

This repository was created to fulfil the **Crixsoft Solution AI Internship**
Project 1 requirement: *Facial Recognition Employee Attendance System*.
Per the internship instructions, the source code is uploaded to a GitHub
repository named `Crixsoft Solution_Facial-Recognition-Attendance-System`,
with the project completion shared on LinkedIn (mentioning **@Crixsoft
Solution**) along with a short video walkthrough and a link to this repo.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Crixsoft Solution — [www.crixsoftsolution.com](http://www.crixsoftsolution.com) · info@crixsoftsolution.com
