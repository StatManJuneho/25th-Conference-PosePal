# import libraries
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


landmarks = ['class', 'posture_type']
for val in range(1, 33 + 1):
    landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

# CSV 파일 초기화
output_csv = 'lunge_labeling.csv'
with open(output_csv, mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

def export_landmark(results, filename, action_label, posture_type):
    try:
        keypoints = [filename, action_label, posture_type] + [
            coord for res in results.pose_landmarks.landmark for coord in [res.x, res.y, res.z, res.visibility]
        ]
        with open(output_csv, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        print(f"Error: {e}")

# 이미지 경로
folder = "/root/posepal/lunge/"
image_files = sorted([file for file in os.listdir(folder) if file.endswith('.jpg')])

# 이미지 순회 및 랜드마크 추출
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for file in image_files:
            file_path = os.path.join(folder, file)
                    
            # Action Label: 파일명 앞 3자리 숫자 추출
            try:
                action_label = file.split('-')[0]
            except IndexError:
                print(f"Inavlid file name format: {file}")
                continue
                    
            # Posture Type 결정
            posture_type = 'correct' if action_label == '081' else 'incorrect'
                    
            # 이미지 읽기
            image = cv2.imread(file_path)
            if image is None:
                print(f"Could not read image: {file}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
            # Mediapipe로 Pose 추정
            results = pose.process(image_rgb)
                    
            # Landmarks 추출 및 CSV 저장
            if results.pose_landmarks:
                export_landmark(results, file, action_label, posture_type)
                print(f"Processed: {file} | Label: {action_label} | Type: {posture_type}")

