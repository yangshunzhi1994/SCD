# coding: utf-8
import numpy as np
import cv2
import os

pwd = 'test/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)

pwd = 'test/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)
    
pwd = 'test/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)

pwd = 'test/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    cv2.rectangle(img, (0, 50),(50, 100), (0, 0, 0), -1)
    occlusion_pwd = 'occlusion/left_lower/' + pwd
    if not os.path.isdir(occlusion_pwd):
        os.makedirs(occlusion_pwd)
    occlusion_filename = os.path.join(occlusion_pwd, filename)
    cv2.imwrite(occlusion_filename, img)