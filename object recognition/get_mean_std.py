import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils
from datasets.MetaRAF import MetaRAF
from datasets.MetaPET import MetaPET
from datasets.MetaFairFace import MetaFairFace
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CNN Training')
parser.add_argument('--dataset', type=str, default='FairFace', help='RAF,PET,FairFace')
parser.add_argument('--train_bs', default=64, type=int, help='learning rate')
parser.add_argument('--test_bs', default=16, type=int, help='learning rate')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(92),
    transforms.RandomHorizontalFlip(),
])

teacher_norm = transforms.Compose([
    transforms.ToTensor(),
])

student_norm = transforms.Compose([
    transforms.Resize(44),
    transforms.ToTensor(),
])

teacher_test = transforms.Compose([
    transforms.TenCrop(92),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
student_test = transforms.Compose([
    transforms.Resize(48),
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

if opt.dataset  == 'RAF':
    print('This is RAF..')
    trainset = MetaRAF(split = 'train', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    validset = MetaRAF(split = 'valid', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    testset = MetaRAF(split = 'test', transform=None, student_norm=student_test, teacher_norm=teacher_test)
elif opt.dataset  == 'PET':
    print('This is PET..')
    trainset = MetaPET(split = 'train', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    validset = MetaPET(split = 'valid', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    testset = MetaPET(split = 'test', transform=None, student_norm=student_test, teacher_norm=teacher_test)
elif opt.dataset  == 'FairFace':
    print('This is FairFace..')
    trainset = MetaFairFace(split = 'train', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    validset = MetaFairFace(split = 'valid', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
    testset = MetaFairFace(split = 'test', transform=None, student_norm=student_test, teacher_norm=teacher_test)
else:
    raise Exception('Invalid dataset name...')
    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=1)
validloader = torch.utils.data.DataLoader(validset, batch_size=opt.train_bs, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.test_bs, shuffle=True, num_workers=1)

train_mean=0 
train_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (inputs, _, targets, _) in enumerate(trainloader):
        train_mean += np.mean(inputs.numpy(), axis=(0,2,3))
        train_std += np.std(inputs.numpy(), axis=(0,2,3))
        mean = train_mean/(batch_idx+1)
        std = train_std/(batch_idx+1)    
    train_mean=0 
    train_std=0
    epoch_mean += mean
    epoch_std += std
print('------train_teacher--------')
print (epoch_mean/epoch, epoch_std/epoch)



train_mean=0 
train_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (_, inputs, targets, _) in enumerate(trainloader):
        train_mean += np.mean(inputs.numpy(), axis=(0,2,3))
        train_std += np.std(inputs.numpy(), axis=(0,2,3))
        mean = train_mean/(batch_idx+1)
        std = train_std/(batch_idx+1)    
    train_mean=0 
    train_std=0
    epoch_mean += mean
    epoch_std += std
print('------train_student--------')
print (epoch_mean/epoch, epoch_std/epoch)



valid_mean=0 
valid_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (inputs, _, targets, _) in enumerate(validloader):
        valid_mean += np.mean(inputs.numpy(), axis=(0,2,3))
        valid_std += np.std(inputs.numpy(), axis=(0,2,3))
        mean = valid_mean/(batch_idx+1)
        std = valid_std/(batch_idx+1)    
    valid_mean=0 
    valid_std=0
    epoch_mean += mean
    epoch_std += std
print('------valid_teacher--------')
print (epoch_mean/epoch, epoch_std/epoch)



valid_mean=0 
valid_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (_, inputs, targets, _) in enumerate(validloader):
        valid_mean += np.mean(inputs.numpy(), axis=(0,2,3))
        valid_std += np.std(inputs.numpy(), axis=(0,2,3))
        mean = valid_mean/(batch_idx+1)
        std = valid_std/(batch_idx+1)    
    valid_mean=0 
    valid_std=0
    epoch_mean += mean
    epoch_std += std
print('------valid_student--------')
print (epoch_mean/epoch, epoch_std/epoch)












test_mean=0 
test_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (inputs, _, targets) in enumerate(testloader):
        test_mean += np.mean(inputs.numpy(), axis=(0,1,3,4))
        test_std += np.std(inputs.numpy(), axis=(0,1,3,4))
        mean = test_mean/(batch_idx+1)
        std = test_std/(batch_idx+1)    
    test_mean=0 
    test_std=0
    epoch_mean += mean
    epoch_std += std
print('------test_teacher---------')
print (epoch_mean/epoch, epoch_std/epoch)



test_mean=0 
test_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 10):
    for batch_idx, (_, inputs, targets) in enumerate(testloader):
        test_mean += np.mean(inputs.numpy(), axis=(0,1,3,4))
        test_std += np.std(inputs.numpy(), axis=(0,1,3,4))
        mean = test_mean/(batch_idx+1)
        std = test_std/(batch_idx+1)    
    test_mean=0 
    test_std=0
    epoch_mean += mean
    epoch_std += std
print('------test_student---------')
print (epoch_mean/epoch, epoch_std/epoch)