import os
import shutil
import numpy as np
import h5py
import skimage.io
import cv2 as cv
import random


# with open("../datasets/ijbc_face_tid_mid.txt", 'r') as f:
#     for line in f.readlines()[0:]:
#         pair = line.strip().split()
#         f_src = "../datasets/IJBC/" + pair[0]
#         dir_path = "../datasets/IJBC/" + pair[1] + "/"
#         if not os.path.isdir(dir_path):
#             os.makedirs(dir_path)
#         f_dst = dir_path + pair[0]
#         shutil.move(f_src, f_dst)


# dir_names = os.listdir("../datasets/IJBC/")
# dir_names.sort()
# dir_names.sort(key=lambda x: int(x.split('_')[0][:6]))
# lable = 0
# for fi in dir_names:
#     src = "../datasets/IJBC/" + fi
#     dst = "../datasets/IJBC/" + str(lable)
#     os.rename(src, dst)
#     lable = lable + 1


# datasets_path = "../datasets/IJBC/"
# types_name = os.listdir(datasets_path)
# types_name.sort()
# types_name.sort(key=lambda x: int(x.split('_')[0][:6]))
# aa = skimage.io.imread('../datasets/2.jpg')
# img_aa = cv.resize(aa, (112, 112))
# for fi in types_name[:5000]:
#     print(fi)
#     train_data_x = []
#     train_data_y = []
#     path = datasets_path + fi + "/"
#     files_names = os.listdir(path)
#     for filename in files_names:
#         file_path = datasets_path + fi + "/" + filename
#         I = skimage.io.imread(os.path.join(file_path))
#         I = cv.resize(I, (112, 112))
#         if len(I.shape) != 3:
#             img2 = np.zeros_like(img_aa)
#             img2[:, :, 0] = I
#             img2[:, :, 1] = I
#             img2[:, :, 2] = I
#             I = img2
#         train_data_x.append(I.tolist())
#         train_data_y.append(int(fi))
#
#     print(np.shape(train_data_x))
#     print(np.shape(train_data_y))
#     datafile = h5py.File("../datasets/IJBC_data_" + str(fi) + ".h5", 'w')
#     datafile.create_dataset("train_data_pixel", dtype='uint8', data=train_data_x)
#     datafile.create_dataset("train_data_label", dtype='int64', data=train_data_y)
#     datafile.close()
#     print("Save data finish!!!")












# pairs_path = "IJBC_test_pair.txt"
# f = open(pairs_path, 'a')
# dir_names = os.listdir("../datasets/IJBC/")
# for fi in dir_names:
#     file_dirs = "../datasets/IJBC/" + fi + "/"
#     file_dirs = os.listdir(file_dirs)
#     if len(file_dirs) > 4:
#         picture_indexes = np.random.choice(range(0, len(file_dirs) - 1), 4)
#         file_dirs = [str(file_dirs[picture_indexes[0]]), str(file_dirs[picture_indexes[1]]),
#                      str(file_dirs[picture_indexes[2]]), str(file_dirs[picture_indexes[3]])]
#         for file_name0 in file_dirs:
#             for file_name1 in file_dirs:
#                 if file_name0 != file_name1:
#                     f.write("%s  %s  %s\n" % (fi, file_name0, file_name1))
#     else:
#         for file_name0 in file_dirs:
#             for file_name1 in file_dirs:
#                 if file_name0 != file_name1:
#                     f.write("%s  %s  %s\n" % (fi, file_name0, file_name1))
#
# for fi in dir_names:
#     file_dirs = "../datasets/IJBC/" + fi + "/"
#     file_dirs = os.listdir(file_dirs)
#     if len(file_dirs) > 4:
#         picture_indexes = np.random.choice(range(0, len(file_dirs) - 1), 4)
#         file_dirs = [str(file_dirs[picture_indexes[0]]), str(file_dirs[picture_indexes[1]]),
#                      str(file_dirs[picture_indexes[2]]), str(file_dirs[picture_indexes[3]])]
#     for file_name0 in file_dirs:
#         dir_indexes = np.random.choice(range(22000, 22000 + len(dir_names)-1), 4)
#         while fi in dir_indexes:
#             dir_indexes = np.random.choice(range(22000, 22000 + len(dir_names)-1), 4)
#         for dir_indexe in dir_indexes:
#             dir_name1 = "../datasets/IJBC/" + str(dir_indexe) + "/"
#             dir_name1 = os.listdir(dir_name1)
#             c = random.randint(0, len(dir_name1) - 1)
#             file_name1 = dir_name1[c]
#             f.write("%s  %s  %s  %s\n" % (fi, file_name0, dir_indexe, file_name1))





































lfw_dir = "../datasets/IJBC"
pairs_path = "../datasets/IJBC_test_pair.txt"


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

datafile = h5py.File("../datasets/IJBC_test_data.h5", 'w')
datafile.create_dataset("train_data_pixel1", dtype='uint8', data=train_data_x1)
datafile.create_dataset("train_data_pixel2", dtype='uint8', data=train_data_x2)
datafile.create_dataset("train_data_label", dtype='bool', data=train_data_y)
datafile.close()

print("Save data finish!!!")

LFW112 = h5py.File("../datasets/IJBC_test_data.h5", 'r')
print(LFW112['train_data_label'][0])
print(LFW112['train_data_pixel1'][0])