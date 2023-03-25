# coding: utf-8
import numpy as np
import cv2
import os

def eye_cascade(img, scaleFactor, minNeighbors=3):
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')   #眼睛
    eye_cascade.load('haarcascades/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, maxSize=(30, 30))
    return eyes
def left_eye_cascade(img, scaleFactor, minNeighbors=3):
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')   #左眼睛
    eye_cascade.load('haarcascades/haarcascade_lefteye_2splits.xml')
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, maxSize=(30, 30))
    return eyes
def right_eye_cascade(img, scaleFactor, minNeighbors=3):
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')   #右眼睛
    eye_cascade.load('haarcascades/haarcascade_righteye_2splits.xml')
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, maxSize=(30, 30))
    return eyes

def cycle_cascade(pwd, filename):
    number = 31
    scaleFactor = 1.005
    img = cv2.imread(os.path.join(pwd, filename))
    eyes = eye_cascade(img, scaleFactor)
    if len(eyes) > 0:
        all_eyes = eyes
        not_eyes = 0
    else:
        not_eyes = 1
        all_eyes = eyes
    for i in range(1, number):
        if scaleFactor < 10.0:
            eyes = eye_cascade(img, scaleFactor, minNeighbors=i)
            if len(eyes) > 0:
                if not_eyes == 1:
                    all_eyes = eyes
                    not_eyes = 0
                else:
                    all_eyes = np.vstack((all_eyes,eyes))
            eyes = left_eye_cascade(img, scaleFactor, minNeighbors=i)
            if len(eyes) > 0:
                if not_eyes == 1:
                    all_eyes = eyes
                    not_eyes = 0
                else:
                    all_eyes = np.vstack((all_eyes,eyes))
            eyes = right_eye_cascade(img, scaleFactor, minNeighbors=i)
            if len(eyes) > 0:
                if not_eyes == 1:
                    all_eyes = eyes
                    not_eyes = 0
                else:
                    all_eyes = np.vstack((all_eyes,eyes))
            scaleFactor = scaleFactor + 0.005
    if len(all_eyes) > 0:
        return img, all_eyes
    else:
        return img, eyes

pwd = 'test/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, eyes = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in eyes:
        if ey > 30:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/eye/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)