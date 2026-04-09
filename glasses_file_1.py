# libraries --------------------------------

import cv2 as cv 
import mediapipe as mp 
import numpy as np 
import pygame as py 
import pyttsx3 as tts  
import speech_recognition as sr 
from gtts import gTTs 
import time 
import random
import datetime
import threading 

#------------------------------------------

#intializing the window -------------------

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame.resize(700, 500)
    cv.imshow("glasses", frame) 

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
