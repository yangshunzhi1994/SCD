''' FairFace-DB Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import os
import cv2
import random
import torchvision
from torchvision import transforms as transforms
import utils  
        
class MetaFairFace(data.Dataset):
    def __init__(self, split='train', transform=None, student_norm=None, teacher_norm=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('datasets/FairFace_MetaData_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'train':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((75790, 100, 100, 3))
        
        elif self.split == 'valid':
            self.valid_data = self.data['valid_data_pixel']
            self.valid_labels = self.data['valid_data_label']
            self.valid_data = np.asarray(self.valid_data)
            self.valid_data = self.valid_data.reshape((10954, 100, 100, 3))
            
        else:
            self.test_data = self.data['test_data_pixel']
            self.test_labels = self.data['test_data_label']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((10954, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'train':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target, index
        
        elif self.split == 'valid':
            img, target = self.valid_data[index], self.valid_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target, index

        else:
            img, target = self.test_data[index], self.test_labels[index]

            img = Image.fromarray(img)
            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'valid':
            return len(self.valid_data)
        else:
            return len(self.test_data)
        
        
        
        
        
class FairFace_teacher(data.Dataset):
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split 
        self.data = h5py.File('datasets/FairFace_data_100.h5', 'r', driver='core')
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((86744, 100, 100, 3))

        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((10954, 100, 100, 3))

    def __getitem__(self, index):
        
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)
        
        
class FairFace_Online(data.Dataset):
    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('datasets/FairFace_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((86744, 100, 100, 3))
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((10954, 100, 100, 3))

    def __getitem__(self, index):
        
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target, index

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)