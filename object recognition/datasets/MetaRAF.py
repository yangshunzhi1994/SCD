''' RAF-DB Dataset class'''

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
        
class MetaRAF(data.Dataset):
    def __init__(self, split='train', transform=None, student_norm=None, teacher_norm=None, noise=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('datasets/RAF_MetaData_100.h5', 'r', driver='core')
        
        self.noise = noise
        if self.noise == 'eye':
            self.data_noise = h5py.File('datasets/RAF_MetaData_eye_100.h5', 'r', driver='core')
        elif self.noise == 'horizontal':
            self.data_noise = h5py.File('datasets/RAF_MetaData_horizontal_100.h5', 'r', driver='core')
        elif self.noise == 'left_lower':
            self.data_noise = h5py.File('datasets/RAF_MetaData_left_lower_100.h5', 'r', driver='core')
        elif self.noise == 'left_upper':
            self.data_noise = h5py.File('datasets/RAF_MetaData_left_upper_100.h5', 'r', driver='core')
        elif self.noise == 'mouth':
            self.data_noise = h5py.File('datasets/RAF_MetaData_mouth_100.h5', 'r', driver='core')
        elif self.noise == 'right_lower':
            self.data_noise = h5py.File('datasets/RAF_MetaData_right_lower_100.h5', 'r', driver='core')
        elif self.noise == 'right_upper':
            self.data_noise = h5py.File('datasets/RAF_MetaData_right_upper_100.h5', 'r', driver='core')
        elif self.noise == 'vertical':
            self.data_noise = h5py.File('datasets/RAF_MetaData_vertical_100.h5', 'r', driver='core')
        elif self.noise == 'AverageBlur':
            self.data_noise = h5py.File('datasets/RAF_MetaData_AverageBlur_100.h5', 'r', driver='core') 
        elif self.noise == 'BilateralFilter':
            self.data_noise = h5py.File('datasets/RAF_MetaData_BilateralFilter_100.h5', 'r', driver='core')
        elif self.noise == 'GaussianBlur':
            self.data_noise = h5py.File('datasets/RAF_MetaData_GaussianBlur_100.h5', 'r', driver='core')
        elif self.noise == 'MedianBlur':
            self.data_noise = h5py.File('datasets/RAF_MetaData_MedianBlur_100.h5', 'r', driver='core')
        elif self.noise == 'Salt-and-pepper':
            self.data_noise = h5py.File('datasets/RAF_MetaData_Salt-and-pepper_100.h5', 'r', driver='core')
        else:
            pass
        
        # now load the picked numpy arrays
        if self.split == 'train':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((9203, 100, 100, 3))
            if self.noise != 'none':
                self.train_data_noise = self.data_noise['train_data_pixel']
                self.train_labels_noise = self.data_noise['train_data_label']
                self.train_data_noise = np.asarray(self.train_data_noise)
                self.train_data_noise = self.train_data_noise.reshape((9203, 100, 100, 3))
        
        elif self.split == 'valid':
            self.valid_data = self.data['valid_data_pixel']
            self.valid_labels = self.data['valid_data_label']
            self.valid_data = np.asarray(self.valid_data)
            self.valid_data = self.valid_data.reshape((3068, 100, 100, 3))
            if self.noise != 'none':
                self.valid_data_noise = self.data_noise['valid_data_pixel']
                self.valid_labels_noise = self.data_noise['valid_data_label']
                self.valid_data_noise = np.asarray(self.valid_data_noise)
                self.valid_data_noise = self.valid_data_noise.reshape((3068, 100, 100, 3))
            
        else:
            self.test_data = self.data['test_data_pixel']
            self.test_labels = self.data['test_data_label']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((3068, 100, 100, 3))
            if self.noise != 'none':
                self.test_data_noise = self.data_noise['test_data_pixel']
                self.test_labels_noise = self.data_noise['test_data_label']
                self.test_data_noise = np.asarray(self.test_data_noise)
                self.test_data_noise = self.test_data_noise.reshape((3068, 100, 100, 3))

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
            
            if self.noise != 'none':
                img_noise, target_noise = self.train_data_noise[index], self.train_labels_noise[index]
                img_noise = Image.fromarray(img_noise)
                img_noise = self.transform(img_noise)
                img_student_noise = self.student_norm(img_noise)
                return img_teacher, img_student_noise, target, index
            else:
                return img_teacher, img_student, target, index
        
        elif self.split == 'valid':
            img, target = self.valid_data[index], self.valid_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
            
            if self.noise != 'none':
                img_noise, target_noise = self.valid_data_noise[index], self.valid_labels_noise[index]
                img_noise = Image.fromarray(img_noise)
                img_noise = self.transform(img_noise)
                img_student_noise = self.student_norm(img_noise)
                return img_teacher, img_student_noise, target, index
            else:
                return img_teacher, img_student, target, index

        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(img)
            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
            
            if self.noise != 'none':
                img_noise, target_noise = self.test_data_noise[index], self.test_labels_noise[index]
                img_noise = Image.fromarray(img_noise)
                img_student_noise = self.student_norm(img_noise)
                return img_teacher, img_student_noise, target
            else:
                return img_teacher, img_student, target

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'valid':
            return len(self.valid_data)
        else:
            return len(self.test_data)
        
        
        
        
        
class RAF_teacher(data.Dataset):
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split 
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))

        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

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
        
        
class RAF_Online(data.Dataset):
    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None, noise=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        self.noise = noise
        if self.noise == 'eye':
            self.data_noise = h5py.File('datasets/RAF_data_eye_100.h5', 'r', driver='core')
        elif self.noise == 'horizontal':
            self.data_noise = h5py.File('datasets/RAF_data_horizontal_100.h5', 'r', driver='core')
        elif self.noise == 'left_lower':
            self.data_noise = h5py.File('datasets/RAF_data_left_lower_100.h5', 'r', driver='core')
        elif self.noise == 'left_upper':
            self.data_noise = h5py.File('datasets/RAF_data_left_upper_100.h5', 'r', driver='core')
        elif self.noise == 'mouth':
            self.data_noise = h5py.File('datasets/RAF_data_mouth_100.h5', 'r', driver='core')
        elif self.noise == 'right_lower':
            self.data_noise = h5py.File('datasets/RAF_data_right_lower_100.h5', 'r', driver='core')
        elif self.noise == 'right_upper':
            self.data_noise = h5py.File('datasets/RAF_data_right_upper_100.h5', 'r', driver='core')
        elif self.noise == 'vertical':
            self.data_noise = h5py.File('datasets/RAF_data_vertical_100.h5', 'r', driver='core')
        elif self.noise == 'AverageBlur':
            self.data_noise = h5py.File('datasets/RAF_data_AverageBlur_100.h5', 'r', driver='core')
        elif self.noise == 'BilateralFilter':
            self.data_noise = h5py.File('datasets/RAF_data_BilateralFilter_100.h5', 'r', driver='core')
        elif self.noise == 'GaussianBlur':
            self.data_noise = h5py.File('datasets/RAF_data_GaussianBlur_100.h5', 'r', driver='core')
        elif self.noise == 'MedianBlur':
            self.data_noise = h5py.File('datasets/RAF_data_MedianBlur_100.h5', 'r', driver='core')
        elif self.noise == 'Salt-and-pepper':
            self.data_noise = h5py.File('datasets/RAF_data_Salt-and-pepper_100.h5', 'r', driver='core')
        else:
            pass
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))
            
            if self.noise != 'none':
                self.train_data_noise = self.data_noise['train_data_pixel']
                self.train_labels_noise = self.data_noise['train_data_label']
                self.train_data_noise = np.asarray(self.train_data_noise)
                self.train_data_noise = self.train_data_noise.reshape((12271, 100, 100, 3))
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))
            
            if self.noise != 'none':
                self.PrivateTest_data_noise = self.data_noise['valid_data_pixel']
                self.PrivateTest_labels_noise = self.data_noise['valid_data_label']
                self.PrivateTest_data_noise = np.asarray(self.PrivateTest_data_noise)
                self.PrivateTest_data_noise = self.PrivateTest_data_noise.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
            
            if self.noise != 'none':
                img_noise, target_noise = self.train_data_noise[index], self.train_labels_noise[index]
                img_noise = Image.fromarray(img_noise)
                img_noise = self.transform(img_noise)
                img_student_noise = self.student_norm(img_noise)
                
                return img_teacher, img_student_noise, target, index
                
            else:
                return img_teacher, img_student, target, index

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
            
            if self.noise != 'none':
                img_noise, target_noise = self.PrivateTest_data_noise[index], self.PrivateTest_labels_noise[index]
                img_student_noise = self.student_norm(img_noise)
                return img_teacher, img_student_noise, target
            else:
                return img_teacher, img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)