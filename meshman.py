import mediapipe as mp
import cv2
import numpy as np
import os

mp_face_mesh= mp.solutions.face_mesh

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))



def calculate_EAR(eye_landmarks):
    A=euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B=euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C=euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    EAR=(A+B)/(2.0*C)
    return EAR


face_mesh=mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,refine_landmarks=True)




LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read.")
        return None
    rgb_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks=results.multi_face_landmarks[0].landmark
        h,w=image.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        left_eye=[points[i] for i in LEFT_EYE_INDICES]
        right_eye=[points[i] for i in RIGHT_EYE_INDICES]    
        left_EAR=calculate_EAR(left_eye)
        right_EAR=calculate_EAR(right_eye)
        EAR=(left_EAR + right_EAR) / 2.0
        return EAR
    _, tresh= cv2.threshold(image, 30,255, cv2.THRESH_BINARY_INV)
    y,x=np.where(tresh>0)
    if len(x) == 0 or len(y) == 0:
        print("❌ Could not estimate geometric EAR (no contours)")
        return None

    height = y.max() - y.min()
    width = x.max() - x.min()
    EAR_geo = height / width if width > 0 else None

    print(f"{image_path}: EAR (geometric) = {EAR_geo:.3f}" if EAR_geo else "Geometric EAR failed")
    return EAR_geo





DATA_DIRS = [
    "dataset/train",
    "dataset/val",
    "dataset/test"
]

for dir_path in DATA_DIRS:
    print(f"\nProcessing folder: {dir_path}")
    for file_name in os.listdir(dir_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(dir_path, file_name)
            process_image(image_path)