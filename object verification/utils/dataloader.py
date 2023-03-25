import numpy as np
from PIL import Image
import h5py
from torch.utils.data.dataset import Dataset
from torchvision import transforms as transforms

class FacenetDataset(Dataset):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.data = h5py.File('../datasets/IJBC_data.h5', 'r') # (439614, 112, 112, 3)
        self.train_data = self.data['train_data_pixel']
        self.train_labels = self.data['train_data_label']
        self.train_data = np.asarray(self.train_data)
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.input_shape[0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        images, labels = self.train_data[index], self.train_labels[index]
        images = Image.fromarray(images)
        images = self.transform(images)
        return images, labels, index



class LFWDataset(Dataset):
    def __init__(self, image_size, data_name):
        self.data_name = data_name
        if self.data_name == "IJBC":
            self.data = h5py.File('../datasets/IJBC_test_data.h5', 'r')  # [27770, 112, 112, 3]
        elif self.data_name == "LFW":
            self.data = h5py.File('../datasets/LFW_test_data.h5', 'r')  # [6000, 112, 112, 3]
        elif self.data_name == "SCface":
            self.data = h5py.File('../datasets/SCface_test_data.h5', 'r')  # [131560, 112, 112, 3]
        elif self.data_name == "Tinyface":
            self.data = h5py.File('../datasets/Tinyface_test_data.h5', 'r')  # [121162, 112, 112, 3]
        self.train_data1 = self.data['train_data_pixel1']
        self.train_data2 = self.data['train_data_pixel2']
        self.train_labels = self.data['train_data_label']
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image1, image2, issame = self.train_data1[index], self.train_data2[index], self.train_labels[index]
        image1 = Image.fromarray(image1)
        image1 = self.transform(image1)
        image2 = Image.fromarray(image2)
        image2 = self.transform(image2)
        return image1, image2, issame

    def __len__(self):
        return len(self.train_data1)