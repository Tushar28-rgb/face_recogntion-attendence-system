import os
import csv
from datetime import datetime

import cv2
import dlib
import numpy as np


SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "models/dlib_face_recognition_resnet_model_v1.dat"
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.6
FRAME_RESIZE_WIDTH = 640


def load_dlib_models():
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        raise FileNotFoundError(
            f"Missing landmark model: {SHAPE_PREDICTOR_PATH}. "
            f"Download from dlib model zoo and place it in the 'models' folder."
        )
    if not os.path.exists(FACE_RECOGNITION_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing face recognition model: {FACE_RECOGNITION_MODEL_PATH}. "
            f"Download from dlib model zoo and place it in the 'models' folder."
        )

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    return detector, shape_predictor, face_rec_model


def compute_face_descriptor(img, rect, shape_predictor, face_rec_model):
    shape = shape_predictor(img, rect)
    face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)


def load_known_faces(detector, shape_predictor, face_rec_model):
    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        print(f"Created '{KNOWN_FACES_DIR}' directory. Put labeled images inside it.")
        return known_encodings, known_names

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for fname in os.listdir(person_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(person_dir, fname)
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            dets = detector(image_rgb, 1)
            if len(dets) == 0:
                continue

            descriptor = compute_face_descriptor(
                image_rgb, dets[0], shape_predictor, face_rec_model
            )
            known_encodings.append(descriptor)
            known_names.append(person_name)

    print(f"Loaded {len(known_encodings)} known faces.")
    return np.array(known_encodings), known_names


def recognize_face(encoding, known_encodings, known_names, tolerance=TOLERANCE):
    if len(known_encodings) == 0:
        return "Unknown"

    distances = np.linalg.norm(known_encodings - encoding, axis=1)
    idx = np.argmin(distances)
    if distances[idx] <= tolerance:
        return known_names[idx]
    return "Unknown"


def init_attendance_file():
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    if not file_exists:
        with open(ATTENDANCE_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "datetime"])


def mark_attendance_once_per_session(name, seen_names):
    if name == "Unknown":
        return
    if name in seen_names:
        return

    seen_names.add(name)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, now])
    print(f"Marked attendance for {name} at {now}")


def run_realtime_recognition():
    detector, shape_predictor, face_rec_model = load_dlib_models()
    known_encodings, known_names = load_known_faces(detector, shape_predictor, face_rec_model)

    init_attendance_file()
    seen_names = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if w > FRAME_RESIZE_WIDTH:
                scale = FRAME_RESIZE_WIDTH / w
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(rgb_frame, 0)

            for rect in dets:
                encoding = compute_face_descriptor(
                    rgb_frame, rect, shape_predictor, face_rec_model
                )
                name = recognize_face(encoding, known_encodings, known_names)

                mark_attendance_once_per_session(name, seen_names)

                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Dlib Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_recognition()

