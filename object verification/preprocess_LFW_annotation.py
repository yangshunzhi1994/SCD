import os
import numpy as np
import h5py
import skimage.io
import cv2 as cv


lfw_dir = "../datasets/LFW"
pairs_path = "../datasets/LFW/lfw_pair.txt"
image_size = [112, 112, 3]


def read_lfw_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)

file_ext="jpg"
pairs = read_lfw_pairs(pairs_path)
nrof_skipped_pairs = 0
path_list = []

for i in range(len(pairs)):
#for pair in pairs:
    pair = pairs[i]
    if len(pair) == 3:
        path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
        path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
        issame = True
    elif len(pair) == 4:
        path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
        path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
        issame = False
    if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
        path_list.append((path0,path1, issame))
    else:
        nrof_skipped_pairs += 1
if nrof_skipped_pairs>0:
    print('Skipped %d image pairs' % nrof_skipped_pairs)

aa = skimage.io.imread('../datasets/2.jpg')
img_aa = cv.resize(aa, (112, 112))
train_data_x1 = []
train_data_x2 = []
train_data_y = []
for i in range(0, len(path_list)):
    (path_1, path_2, issame) = path_list[i]

    image1 = skimage.io.imread(os.path.join(path_1))
    image1 = cv.resize(image1, (112, 112))
    if len(image1.shape) != 3:
        img2 = np.zeros_like(img_aa)
        img2[:, :, 0] = image1
        img2[:, :, 1] = image1
        img2[:, :, 2] = image1
        image1 = img2
    train_data_x1.append(image1.tolist())

    image2 = skimage.io.imread(os.path.join(path_2))
    image2 = cv.resize(image2, (112, 112))
    if len(image2.shape) != 3:
        img2 = np.zeros_like(img_aa)
        img2[:, :, 0] = image2
        img2[:, :, 1] = image2
        img2[:, :, 2] = image2
        image2 = img2
    train_data_x2.append(image2.tolist())

    train_data_y.append(issame)

print(np.shape(train_data_x1))
print(np.shape(train_data_x2))
print(np.shape(train_data_y))

datafile = h5py.File("../datasets/LFW_data.h5", 'w')
datafile.create_dataset("train_data_pixel1", dtype='uint8', data=train_data_x1)
datafile.create_dataset("train_data_pixel2", dtype='uint8', data=train_data_x2)
datafile.create_dataset("train_data_label", dtype='bool', data=train_data_y)
datafile.close()

print("Save data finish!!!")

LFW112 = h5py.File("../datasets/LFW_data.h5", 'r')
print(LFW112['train_data_label'][0])
print(LFW112['train_data_pixel1'][0])