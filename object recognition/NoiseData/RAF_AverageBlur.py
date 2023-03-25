# coding: utf-8
import numpy as np
import cv2
import os

pwd = 'test/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
    
    
    
    
    
    
    
    
pwd = 'train/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
    
    
    
    
    
    
    
    
    
pwd = 'valid/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = cv2.blur(img,(5,5))
    noise_pwd = 'Noise/AverageBlur/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)