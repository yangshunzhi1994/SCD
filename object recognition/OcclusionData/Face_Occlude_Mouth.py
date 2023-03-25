# coding: utf-8
import numpy as np
import cv2
import os

def mouth_cascade(img, scaleFactor, minNeighbors=3):
    mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')   #嘴巴
    mouth_cascade.load('haarcascades/haarcascade_mcs_mouth.xml')
    mouths = mouth_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, maxSize=(50, 50))
    return mouths

def cycle_cascade(pwd, filename):
    number = 31
    scaleFactor = 1.005
    img = cv2.imread(os.path.join(pwd, filename))
    all_mouths = []
    for i in range(1, number):
        if scaleFactor < 10.0:
            mouths = mouth_cascade(img, scaleFactor, minNeighbors=i)
            if len(mouths) > 0:
                if len(all_mouths) == 0:
                    all_mouths = mouths
                else:
                    all_mouths = np.vstack((all_mouths,mouths))
            scaleFactor = scaleFactor + 0.005
    if len(all_mouths) > 0:
        return img, all_mouths
    else:
        return img, mouths
    
pwd = 'test/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)

pwd = 'test/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img, mouths = cycle_cascade(pwd, filename)
    for (ex,ey,ew,eh) in mouths:
        if ey < 60:
            pass
        else:
            cv2.rectangle(img, (ex, ey),(ex+ew, ey+eh), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/mouth/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)