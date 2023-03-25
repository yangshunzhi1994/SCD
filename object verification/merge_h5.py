import h5py
import numpy as np

path_H5 = "../datasets/"
data0 = h5py.File(path_H5 + "IJBC_data_0.h5", 'r')
train_data_x = data0['train_data_pixel']
train_data_y = data0['train_data_label']

for i in range(1, 21999):
    data_path = path_H5 + "IJBC_data_" + str(i) + ".h5"
    print(111111111111111111111)
    print(data_path)
    data = h5py.File(data_path, 'r')
    train_data_x = np.concatenate((train_data_x, data['train_data_pixel']), axis=0)
    train_data_y = np.concatenate((train_data_y, data['train_data_label']), axis=0)
    print(len(train_data_x))


print(train_data_x.shape)
print(train_data_y.shape)
datafile = h5py.File("../datasets/IJBC_data.h5", 'w')
datafile.create_dataset("train_data_pixel", dtype='uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype='int64', data=train_data_y)
datafile.close()
print("Save data finish!!!")

SCface112 = h5py.File("../datasets/IJBC_data.h5", 'r')
print(SCface112['train_data_label'].shape)
print(SCface112['train_data_label'][0])
print(1111111111111111)
print(SCface112['train_data_pixel'].shape)
print(SCface112['train_data_pixel'][0])