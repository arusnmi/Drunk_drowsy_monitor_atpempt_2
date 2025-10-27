import cv2
import numpy as np
import mediapipe as mp
import os
import csv
import json

# =============================
# INITIAL SETUP
# =============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =============================
# HELPER FUNCTIONS
# =============================

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_EAR(eye_landmarks):
    """Compute Eye Aspect Ratio."""
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    if C == 0:
        return None
    ear = (A + B) / (2.0 * C)
    return ear

# =============================
# MAIN IMAGE PROCESSOR
# =============================

def process_image(image_path):
    """Compute EAR, MAR, and HTR from a full face image (safe version)."""
    image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_color is None:
        print(f"⚠️ Could not read {image_path}")
        return None, None, None

    rgb_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        print(f"⚠️ No face detected in {os.path.basename(image_path)}")
        return None, None, None

    h, w = image_color.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # --- EAR ---
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    left_eye = [points[i] for i in LEFT_EYE_INDICES]
    right_eye = [points[i] for i in RIGHT_EYE_INDICES]
    left_EAR = calculate_EAR(left_eye)
    right_EAR = calculate_EAR(right_eye)
    EAR = (left_EAR + right_EAR) / 2.0 if left_EAR and right_EAR else None

    # --- MAR ---
    MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]
    mouth = [points[i] for i in MOUTH_INDICES]
    try:
        A = euclidean_distance(mouth[1], mouth[7])
        B = euclidean_distance(mouth[2], mouth[6])
        C = euclidean_distance(mouth[3], mouth[5])
        D = euclidean_distance(mouth[0], mouth[4])  # horizontal
        MAR = (A + B + C) / (2.0 * D) if D > 0 else None
    except Exception:
        MAR = None

    # --- HTR ---
    LEFT_FACE_IDX = 234
    RIGHT_FACE_IDX = 454
    try:
        left_face = points[LEFT_FACE_IDX]
        right_face = points[RIGHT_FACE_IDX]
        dx = abs(left_face[0] - right_face[0])
        dy = abs(left_face[1] - right_face[1])
        HTR = dy / (dx + 1e-6)
    except Exception:
        HTR = None

    # --- Skip incomplete data safely ---
    if EAR is None or MAR is None or HTR is None:
        print(f"⚠️ Skipping {os.path.basename(image_path)} (incomplete features)")
        return None, None, None

    # --- Safe printing ---
    def fmt(v): 
        return f"{v:.3f}" if v is not None else "None"

    print(f"{os.path.basename(image_path)}: EAR={fmt(EAR)}, MAR={fmt(MAR)}, HTR={fmt(HTR)}")
    return EAR, MAR, HTR


# =============================
# DATA COLLECTION
# =============================

def collect_metrics_from_dir(base_dir):
    """Collect EAR, MAR, HTR from open/closed subfolders."""
    ears, mars, htrs, labels = [], [], [], []
    for label_name, label_value in (("open", 0), ("closed", 1)):
        folder = os.path.join(base_dir, label_name)
        if not os.path.isdir(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue
        for fn in os.listdir(folder):
            if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, fn)
            ear, mar, htr = process_image(path)
            if ear is None or mar is None or htr is None:
                continue
            ears.append(ear)
            mars.append(mar)
            htrs.append(htr)
            labels.append(label_value)
    return np.array(ears), np.array(mars), np.array(htrs), np.array(labels)

# =============================
# THRESHOLD FINDER
# =============================

def find_best_threshold(values, labels, direction="low", steps=500):
    """Find best threshold maximizing F1 score."""
    vals = np.array(values)
    labs = np.array(labels)
    vmin, vmax = vals.min(), vals.max()
    best_f1, best_t = -1.0, (vmin + vmax) / 2
    best_metrics = {}

    thresholds = np.linspace(vmin, vmax, steps)
    for t in thresholds:
        preds = (vals <= t).astype(int) if direction == "low" else (vals >= t).astype(int)
        TP = np.sum((preds == 1) & (labs == 1))
        FP = np.sum((preds == 1) & (labs == 0))
        FN = np.sum((preds == 0) & (labs == 1))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_metrics = {"precision": precision, "recall": recall, "f1": f1,
                            "TP": int(TP), "FP": int(FP), "FN": int(FN)}

    return best_t, best_metrics

# =============================
# TRAINING + EVALUATION
# =============================

def compute_thresholds(root_dataset="dataset"):
    """Compute EAR, MAR, and HTR thresholds from training set."""
    train_dir = os.path.join(root_dataset, "train")
    print(f"Collecting metrics from {train_dir} ...")
    ears, mars, htrs, labels = collect_metrics_from_dir(train_dir)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/train_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ear", "mar", "htr", "label"])
        for e, m, h, l in zip(ears, mars, htrs, labels):
            writer.writerow([e, m, h, l])

    ear_t, ear_m = find_best_threshold(ears, labels, "low")
    mar_t, mar_m = find_best_threshold(mars, labels, "high")
    htr_t, htr_m = find_best_threshold(htrs, labels, "high")

    thresholds = {
        "EAR_threshold": float(ear_t),
        "MAR_threshold": float(mar_t),
        "HTR_threshold": float(htr_t),
        "EAR_metrics": ear_m,
        "MAR_metrics": mar_m,
        "HTR_metrics": htr_m
    }

    with open("outputs/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\n=== TRAIN RESULTS ===")
    print(json.dumps(thresholds, indent=2))
    return thresholds

def evaluate_rules(root_dataset="dataset", thresholds=None):
    """Evaluate EAR/MAR/HTR thresholds on val and test sets."""
    if thresholds is None:
        with open("outputs/thresholds.json") as f:
            thresholds = json.load(f)

    ear_t = thresholds["EAR_threshold"]
    mar_t = thresholds["MAR_threshold"]
    htr_t = thresholds["HTR_threshold"]

    report = {}
    for split in ("val", "test"):
        split_dir = os.path.join(root_dataset, split)
        ears, mars, htrs, labels = collect_metrics_from_dir(split_dir)
        if len(ears) == 0:
            continue

        preds = ((ears <= ear_t) | (mars >= mar_t) | (htrs >= htr_t)).astype(int)

        TP = np.sum((preds == 1) & (labels == 1))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        acc = (TP + TN) / max(1, len(labels))

        report[split] = {"precision": precision, "recall": recall,
                         "f1": f1, "accuracy": acc, "TP": int(TP),
                         "FP": int(FP), "FN": int(FN), "TN": int(TN)}

        print(f"\n=== {split.upper()} RESULTS ===")
        print(json.dumps(report[split], indent=2))

    with open("outputs/report.json", "w") as f:
        json.dump(report, f, indent=2)

# =============================
# RUN PIPELINE
# =============================

if __name__ == "__main__":
    thresholds = compute_thresholds("dataset")
    evaluate_rules("dataset", thresholds)
