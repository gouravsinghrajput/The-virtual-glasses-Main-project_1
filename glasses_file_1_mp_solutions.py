# libraries --------------------------------

from __future__ import annotations
from openwakeword.model import Model 
from unittest import result

import cv2 as cv 
import mediapipe as mp 
import pygame as py 
import pyttsx3 as tts  
import speech_recognition as sr 
# from gtts import gTTs 
import numpy as np 
import time 
import random
import datetime
import threading 
# import pvporcupine as pv 
import sounddevice as sd
# import struct
import pyaudio as pa 



mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils  

hands = mp_hands.Hands(
    max_num_hands = 2,
    min_tracking_confidence = 0.6,
    min_detection_confidence = 0.6
)



cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

             
    frame = cv.flip(frame, 1) 
    frame = cv.resize(frame, (700, 500)) 
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 


    results = hands.process(frame)
    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks: 

            h, w, _ = frame.shape 
            landmark_list = [] 

            for landmark_id in range(21):
                x = int(hand_landmarks.landmark[landmark_id].x * w)
                y = int(hand_landmarks.landmark[landmark_id].y * h) 
                landmark_list.append((x, y)) 
                # print(f'Landmark {landmark_id}: ({x}, {y})')

            
            mp_draw.draw_landmarks(frame,
                                   hand_landmarks, 
                                   mp_hands.HAND_CONNECTIONS)
            
    
    cv.imshow('only hands', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break 

cap.release()
cv.destroyAllWindows()

