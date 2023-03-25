# coding: utf-8
import numpy as np
import cv2
import os
import random
from PIL import Image

def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

pwd = 'test/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'test/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
    
    
    
    
    
    
    
    
pwd = 'train/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'train/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
    
    
    
    
    
    
    
    
    
pwd = 'valid/0'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/1'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/2'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/3'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/4'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/5'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)
    
pwd = 'valid/6'
files = os.listdir(pwd)
files.sort()
for filename in files:
    img = cv2.imread(os.path.join(pwd, filename))
    img = sp_noise(img, prob=0.05)
    noise_pwd = 'Noise/Salt-and-pepper/' + pwd
    if not os.path.isdir(noise_pwd):
        os.makedirs(noise_pwd)
    noise_filename = os.path.join(noise_pwd, filename)
    cv2.imwrite(noise_filename, img)