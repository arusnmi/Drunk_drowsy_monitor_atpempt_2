import mediapipe as mp
import cv2
import numpy as np


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





