import os
import shutil
import numpy as np
import h5py
import skimage.io
import cv2 as cv
import random





# file_names = os.listdir("../datasets/tinyface/Testing_Set/")
# file_names = sorted(file_names)
# for fi in file_names:
#     fi_dir = fi.split("_", 1)
#     print(111111111111)
#     print(fi_dir)
#     path = os.path.join("../datasets/tinyface/" + str(fi_dir[0]) + "/")
#     if not os.path.isdir(path):
#         os.makedirs(path)
#     f_src = os.path.join("../datasets/tinyface/Testing_Set/" + str(fi_dir[0]) + "_" + str(fi_dir[1]))
#     f_dst = os.path.join(path + str(fi_dir[0]) + "_" + str(fi_dir[1]))
#     shutil.move(f_src, f_dst)





# dir_names = os.listdir("../datasets/tinyface/")
# dir_names.sort()
# dir_names.sort(key=lambda x: int(x.split('_')[0][:4]))
# lable = 0
# for fi in dir_names:
#     src = "../datasets/tinyface/" + fi
#     dst = "../datasets/tinyface/" + str(lable)
#     os.rename(src, dst)
#     lable = lable + 1












# pairs_path = "../datasets/Tinyface_test_pair.txt"
# f = open(pairs_path, 'a')
# dir_names = os.listdir("../datasets/Tinyface/")
# for fi in dir_names:
#     file_dirs = "../datasets/Tinyface/" + fi + "/"
#     file_dirs = os.listdir(file_dirs)
#     for file_name0 in file_dirs:
#         for file_name1 in file_dirs:
#             if file_name0 != file_name1:
#                 f.write("%s  %s  %s\n" % (fi, file_name0, file_name1))
#
# for fi in dir_names:
#     file_dirs = "../datasets/Tinyface/" + fi + "/"
#     file_dirs = os.listdir(file_dirs)
#     for file_name0 in file_dirs:
#         dir_indexes = np.random.choice(range(0, 5139), 6)
#         while str(fi) in dir_indexes.astype(str):
#             dir_indexes = np.random.choice(range(0, 5139), 6)
#         for dir_indexe in dir_indexes:
#             dir_name1 = "../datasets/Tinyface/" + str(dir_indexe) + "/"
#             dir_name1 = os.listdir(dir_name1)
#             c = random.randint(0, len(dir_name1) - 1)
#             file_name1 = dir_name1[c]
#             f.write("%s  %s  %s  %s\n" % (fi, file_name0, dir_indexe, file_name1))


















lfw_dir = "../datasets/Tinyface"
pairs_path = "../datasets/Tinyface_test_pair.txt"


def read_lfw_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)


pairs = read_lfw_pairs(pairs_path)
nrof_skipped_pairs = 0
path_list = []

for i in range(len(pairs)):
#for pair in pairs:
    pair = pairs[i]
    if len(pair) == 3:
        path0 = os.path.join(lfw_dir, pair[0], pair[1])
        path1 = os.path.join(lfw_dir, pair[0], pair[2])
        issame = True
    elif len(pair) == 4:
        path0 = os.path.join(lfw_dir, pair[0], pair[1])
        path1 = os.path.join(lfw_dir, pair[2], pair[3])
        issame = False
    if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
        path_list.append((path0, path1, issame))
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

datafile = h5py.File("../datasets/Tinyface_test_data.h5", 'w')
datafile.create_dataset("train_data_pixel1", dtype='uint8', data=train_data_x1)
datafile.create_dataset("train_data_pixel2", dtype='uint8', data=train_data_x2)
datafile.create_dataset("train_data_label", dtype='bool', data=train_data_y)
datafile.close()

print("Save data finish!!!")

SCface112 = h5py.File("../datasets/Tinyface_test_data.h5", 'r')
print(SCface112['train_data_label'][0])
print(SCface112['train_data_pixel1'][0])