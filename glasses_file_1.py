# libraries --------------------------------

from __future__ import annotations
from unittest import result

import cv2 as cv 
import numpy as np 
import pygame as py 
import pyttsx3 as tts  
import speech_recognition as sr 
# from gtts import gTTs 
import time 
import random
import datetime
import threading 


import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 



hand_model_path = 'hand_landmarker.task'
#------------------------------------------

BaseOptions = mp.tasks.BaseOptions 
HandLandmarker = mp.tasks.vision.HandLandmarker 
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions 
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult 
VisionRunningMode = mp.tasks.vision.RunningMode 


latest_result = None 

def print_result(result, output_image, timestamp_ms):
    # print('hand landmarker result: {}'.format(result))
    global latest_result
    latest_result = result


options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = hand_model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = print_result,
    num_hands = 2,
    min_tracking_confidence = 0.6,
    min_hand_detection_confidence = 0.6,
    min_hand_presence_confidence = 0.6)



cap = cv.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.flip(frame, 1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame = cv.resize(frame, (700, 500))

        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame)
        
        timestamp_ms = int(time.time() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for hand in latest_result.hand_landmarks:
                for points in hand:
                    h, w, _ = frame.shape
                    cx = int(points.x * w)
                    cy = int(points.y * h)

                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        cv.imshow('hand tracking', frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()