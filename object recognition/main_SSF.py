'''Train RAF/ExpW with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import itertools
import os
import time
import math
import argparse
import utils
import losses
from copy import deepcopy
from Fuzzy_PID import Fuzzy_PID as Fuzzy_PID
from utils import load_pretrained_model
from datasets.MetaRAF import MetaRAF
from datasets.MetaPET import MetaPET
from datasets.MetaFairFace import MetaFairFace
from torch.autograd import Variable
from network.teacherNet import Teacher
from network.studentNet import CNN_RIS
from tensorboardX import SummaryWriter
from utils import ACC_evaluation

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="MetaStudent", help='MetaStudent')
parser.add_argument('--data_name', type=str, default="RAF", help='RAF,PET,FairFace')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=32, type=int, help='Batch size')
parser.add_argument('--P', default=0.3, type=float, help='P')
parser.add_argument('--I', default=0.1, type=float, help='I')
parser.add_argument('--D', default=0.1, type=float, help='D')
parser.add_argument('--TP', default=0.3, type=float, help='TP')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--preload', type=str, default="False", help='Whether to preload the parameters of the best teacher network')
parser.add_argument('--noise', type=str, default='none', help='none,eye,horizontal,left_lower,left_upper,mouth,right_lower,right_upper,vertical') 
                                    # AverageBlur, BilateralFilter, GaussianBlur, MedianBlur, Salt-and-pepper
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_student_ACC = 0
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

total_epoch = args.epochs

path = os.path.join(args.save_root + args.data_name + '_' + args.model + '_' + str(args.preload) + '_P_' + str(args.P) + '_I_' + str(args.I) + '_D_' + str(args.D)+ '_TP_' + str(args.TP) + '_' + args.noise)
writer = SummaryWriter(log_dir=path)

# Data
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                        '+ str(args.model))
print ('The P is:                                '+ str(args.P))
print ('The I is:                                '+ str(args.I))
print ('The D is:                                '+ str(args.D))
print ('The TP is:                               '+ str(args.TP))
print ('The type of noise used is:                    '+ str(args.noise))

transform_train = transforms.Compose([
    transforms.RandomCrop(92),
    transforms.RandomHorizontalFlip(),
])

if args.data_name == 'RAF':
    NUM_CLASSES = 7
    transforms_teacher_train_Normalize = transforms.Normalize((0.59090257, 0.45846474, 0.40872896),
                 (0.256847, 0.23600867, 0.23555762))
    transforms_student_train_Normalize = transforms.Normalize((0.5909926, 0.45852724, 0.40877178), 
                 (0.25130215, 0.23015141, 0.22988683))
    transforms_teacher_valid_Normalize = transforms.Normalize((0.5808446, 0.4549036, 0.4080347),
                 (0.25791392, 0.23598821, 0.23356254))
    transforms_student_valid_Normalize = transforms.Normalize((0.58096486, 0.45507956, 0.40822345), 
                 (0.25251663, 0.23024549, 0.22795384))
    transforms_teacher_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.58956766, 0.45706737, 0.40717924], std=[0.25477308, 0.2334413, 0.23245202])
             (transforms.ToTensor()(crop)) for crop in crops]))
    transforms_student_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.58992976, 0.45730913, 0.40742418], std=[0.2491213, 0.2276218, 0.22677615])
             (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'PET':
    NUM_CLASSES = 37
    transforms_teacher_train_Normalize = transforms.Normalize((0.48090443, 0.44797224, 0.39722067),
                 (0.26371562, 0.25840747, 0.26663858))
    transforms_student_train_Normalize = transforms.Normalize((0.4807216, 0.44780657, 0.3969885), 
                 (0.24953257, 0.24419859, 0.25236925))
    transforms_teacher_valid_Normalize = transforms.Normalize((0.47614348, 0.43873084, 0.39019287),
                 (0.26422974, 0.25770748, 0.26342714))
    transforms_student_valid_Normalize = transforms.Normalize((0.4761954, 0.43884024, 0.39033425), 
                 (0.25000805, 0.24360888, 0.2491732))
    transforms_teacher_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.48613992, 0.45270622, 0.39577982], std=[0.26570764, 0.26151732, 0.26960006])
             (transforms.ToTensor()(crop)) for crop in crops]))
    transforms_student_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.48611367, 0.45263973, 0.39573705], std=[0.25037044, 0.24645199, 0.25450197])
             (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'FairFace':
    NUM_CLASSES = 7
    transforms_teacher_train_Normalize = transforms.Normalize((0.4906074, 0.3599049, 0.30460462),
                 (0.25141177, 0.21795125, 0.21167599))
    transforms_student_train_Normalize = transforms.Normalize((0.4906108, 0.35990545, 0.30459678), 
                 (0.24705137, 0.21349338, 0.20739764))
    transforms_teacher_valid_Normalize = transforms.Normalize((0.49262354, 0.36139473, 0.30580184),
                 (0.25217605, 0.21932942, 0.2129162))
    transforms_student_valid_Normalize = transforms.Normalize((0.49261856, 0.3614008, 0.3058052), 
                 (0.24784128, 0.21488628, 0.20863612))
    transforms_teacher_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.49167913, 0.36098105, 0.30529523], std=[0.24649838, 0.21503104, 0.20875944])
             (transforms.ToTensor()(crop)) for crop in crops]))
    transforms_student_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
        mean=[0.49202734, 0.36110377, 0.30535242], std=[0.24179104, 0.21022305, 0.20413795])
             (transforms.ToTensor()(crop)) for crop in crops]))
else:
    raise Exception('Invalid dataset name...')

teacher_train_norm = transforms.Compose([
transforms.ToTensor(),
transforms_teacher_train_Normalize,
])

student_train_norm = transforms.Compose([
transforms.Resize(44),
transforms.ToTensor(),
transforms_student_train_Normalize,
])

teacher_valid_norm = transforms.Compose([
transforms.ToTensor(),
transforms_teacher_valid_Normalize,
])

student_valid_norm = transforms.Compose([
transforms.Resize(44),
transforms.ToTensor(),
transforms_student_valid_Normalize,
])

teacher_test_norm = transforms.Compose([
transforms.TenCrop(92),
transforms_teacher_test_Normalize,
])

student_test_norm = transforms.Compose([
transforms.Resize(48),
transforms.TenCrop(44),
transforms_student_test_Normalize,
])

if args.data_name == 'RAF':
    last_loss = torch.zeros(3068).cuda()
    lastlast_loss = torch.zeros(3068).cuda()
    trainset = MetaRAF(split = 'train', transform=transform_train, student_norm=student_train_norm, teacher_norm=teacher_train_norm, noise=args.noise)
    validset = MetaRAF(split = 'valid', transform=transform_train, student_norm=student_valid_norm, teacher_norm=teacher_valid_norm, noise=args.noise)
    testset = MetaRAF(split = 'test', transform=None, student_norm=student_test_norm, teacher_norm=teacher_test_norm, noise=args.noise)
elif args.data_name == 'PET':
    last_loss = torch.zeros(956).cuda()
    lastlast_loss = torch.zeros(956).cuda()
    trainset = MetaPET(split = 'train', transform=transform_train, student_norm=student_train_norm, teacher_norm=teacher_train_norm)
    validset = MetaPET(split = 'valid', transform=transform_train, student_norm=student_valid_norm, teacher_norm=teacher_valid_norm)
    testset = MetaPET(split = 'test', transform=None, student_norm=student_test_norm, teacher_norm=teacher_test_norm)
elif args.data_name == 'FairFace':
    last_loss = torch.zeros(10954).cuda()
    lastlast_loss = torch.zeros(10954).cuda()
    trainset = MetaFairFace(split = 'train', transform=transform_train, student_norm=student_train_norm, teacher_norm=teacher_train_norm)
    validset = MetaFairFace(split = 'valid', transform=transform_train, student_norm=student_valid_norm, teacher_norm=teacher_valid_norm)
    testset = MetaFairFace(split = 'test', transform=None, student_norm=student_test_norm, teacher_norm=teacher_test_norm)
else:
    raise Exception('Invalid dataset name...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=1)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.train_bs, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=1)

tnet = Teacher(num_classes=NUM_CLASSES).cuda()
snet = CNN_RIS(num_classes=NUM_CLASSES).cuda()

if args.preload == 'True':
    tcheckpoint = torch.load(os.path.join('results/' + args.data_name+ '_Teacher','Best_Teacher_model.t7'))
    load_pretrained_model(tnet, tcheckpoint['tnet'])
    try:
        print ('best_Teacher_acc is '+ str(tcheckpoint['test_acc']))  
        best_teacher_ACC = tcheckpoint['test_acc']
    except:
        print ('best_Teacher_acc is '+ str(tcheckpoint['best_PrivateTest_acc'])) 
        best_teacher_ACC = 87.74
else:
    best_teacher_ACC = 0 
    print('-------------The teacher network is trained from scratch--------------------')
print('==> Preparing data..')

criterion = torch.nn.CrossEntropyLoss().cuda()
t_optimizer=optim.SGD(tnet.parameters(),lr=args.lr, momentum=0.9, weight_decay=5e-4)
s_optimizer=optim.SGD(snet.parameters(),lr=args.lr, momentum=0.9, weight_decay=5e-4)

pre_tnet, aft_tnet = None, None

def set_PID():
    SP, SI, SD = args.P, args.I, args.D
    return SP, SI, SD

def valid(epoch):
    meta = deepcopy(snet)
    meta.eval()
    tnet.train()
    SP, SI, SD = set_PID()
    
    if epoch > 1:
        err = last_loss.mean()
        SP, SI, SD = Fuzzy_PID(SP, SI, SD, err)
    
    for batch, (img_teacher, img_student, targets, index) in enumerate(validloader):
        img_teacher = img_teacher.cuda()
        img_student = img_student.cuda()
        targets = targets.cuda() 
        img_teacher, img_student, targets = Variable(img_teacher), Variable(img_student), Variable(targets)
        
        t_optimizer.zero_grad()
        rb1_t, rb2_t, rb3_t, feat_t, mimic_t, out_t = tnet(img_teacher)
        with torch.no_grad():
            rb1_s, rb2_s, rb3_s, feat_s, mimic_s, out_s = meta(img_student)
        sloss = F.cross_entropy(out_s, targets, reduction='none') 
        tloss = F.cross_entropy(out_t, targets, reduction='none')
        
        if epoch == 1:
            u = SP*sloss
            weights = F.softmax(u, dim=0)
            weights.requires_grad=True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            utils.clip_gradient(t_optimizer, 0.1)
            t_optimizer.step()
            last_loss[index] = sloss
            
        elif epoch == 2:
            k = torch.div(sloss + last_loss[index], 2)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k
            
            D_loss = sloss - last_loss[index]
            
            u = SP*sloss + SI*I_loss + SD*D_loss 
            weights = F.softmax(u, dim=0)
            weights.requires_grad=True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            utils.clip_gradient(t_optimizer, 0.1)
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = sloss
        else:   
            k = torch.div(sloss + last_loss[index] + lastlast_loss[index], 3)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k
            
            D_loss = sloss - 2*last_loss[index] + lastlast_loss[index]
            
            u = SP*sloss + SI*I_loss + SD*D_loss 
            weights = F.softmax(u, dim=0)
            weights.requires_grad=True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            utils.clip_gradient(t_optimizer, 0.1)
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = sloss        

def train(epoch):
    global pre_tnet, aft_tnet
    print('\nEpoch: %d' % epoch)
    snet.train()
    tnet.train()
    train_loss = 0
    
    tconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(t_optimizer, current_lr)
        utils.set_lr(s_optimizer, current_lr)
    else:
        current_lr = args.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (img_teacher, img_student, targets, index) in enumerate(trainloader):
        if args.cuda:
            img_teacher = img_teacher.cuda()
            img_student = img_student.cuda()
            targets = targets.cuda()
            
        img_teacher, img_student, targets = Variable(img_teacher), Variable(img_student), Variable(targets)
        
        t_optimizer.zero_grad()
        rb1_t, rb2_t, rb3_t, feat_t, mimic_t, out_t = tnet(img_teacher)
        tloss = F.cross_entropy(out_t, targets, reduction='none')
        if epoch == 0:
            tloss = tloss.mean()
        else:
            with torch.no_grad():
                _, _, _, _, _, pre_t = pre_tnet(img_teacher)
                _, _, _, _, _, aft_t = aft_tnet(img_teacher)
            pre_tloss = F.cross_entropy(pre_t, targets, reduction='none')
            aft_tloss = F.cross_entropy(aft_t, targets, reduction='none')
            P_loss = pre_tloss-aft_tloss
            u = args.TP*P_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad=True
            tloss = torch.dot(tloss, weights)
        tloss.backward()
        utils.clip_gradient(t_optimizer, 0.1)
        t_optimizer.step()
        
        s_optimizer.zero_grad()
        rb1_s, rb2_s, rb3_s, feat_s, mimic_s, out_s = snet(img_student)
        with torch.no_grad():
            _, _, _, _, _, out_t = tnet(img_teacher)
            
        if epoch == 0:
            sloss = losses.KL_divergence(temperature = 20).cuda()(out_t, out_s)
        else:
            sloss = losses.KL_divergence_sample(temperature = 20).cuda()(out_t, out_s, weights.detach())
        sloss.backward()
        utils.clip_gradient(s_optimizer, 0.1)
        s_optimizer.step()
        
        train_loss += sloss.item()
        
        tconf_mat, tacc, tmAP, tF1_score = ACC_evaluation(tconf_mat, out_t, targets, NUM_CLASSES)
        sconf_mat, sacc, smAP, sF1_score = ACC_evaluation(sconf_mat, out_s, targets, NUM_CLASSES)
    return train_loss/(batch_idx+1), 100.*tacc, 100.*tmAP, 100.*tF1_score, 100.*sacc, 100.*smAP, 100.*sF1_score

# def test(epoch):
#     tnet.eval()
#     snet.eval()
    
#     tconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
#     sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
#     PrivateTest_loss = 0

#     for batch_idx, (img_teacher, img_student, targets) in enumerate(testloader):
#         t = time.time()
#         test_bs, ncrops, c, h, w = np.shape(img_teacher)
#         test_bs, ncrops, cs, hs, ws = np.shape(img_student)
#         img_teacher = img_teacher.view(-1, c, h, w)
#         img_student = img_student.view(-1, cs, hs, ws)
        
#         img_teacher, img_student, targets = img_teacher.cuda(), img_student.cuda(), targets.cuda()
#         img_teacher, img_student, targets = Variable(img_teacher), Variable(img_student),Variable(targets)
        
#         with torch.no_grad():
#             _, _, _, _, _, out_t = tnet(img_teacher)
#             _, _, _, _, _, out_s = snet(img_student)
            
#         out_t = out_t.view(test_bs, ncrops, -1).mean(1) 
#         out_s = out_s.view(test_bs, ncrops, -1).mean(1)   
#         tloss = criterion(out_t, targets)
#         sloss = criterion(out_s, targets)
#         PrivateTest_loss += sloss.item()
        
#         tconf_mat, tacc, tmAP, tF1_score = ACC_evaluation(tconf_mat, out_t, targets, NUM_CLASSES)
#         sconf_mat, sacc, smAP, sF1_score = ACC_evaluation(sconf_mat, out_s, targets, NUM_CLASSES)
        
#     return PrivateTest_loss/(batch_idx+1), 100.*tacc, 100.*tmAP, 100.*tF1_score, 100.*sacc, 100.*smAP, 100.*sF1_score


def test(epoch):
    snet.eval()
    sconf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    PrivateTest_loss = 0

    for batch_idx, (_, img_student, targets) in enumerate(testloader):
        t = time.time()
        test_bs, ncrops, cs, hs, ws = np.shape(img_student)
        img_student = img_student.view(-1, cs, hs, ws)
        
        img_student, targets = img_student.cuda(), targets.cuda()
        img_student, targets = Variable(img_student),Variable(targets)
        
        with torch.no_grad():
            _, _, _, _, _, out_s = snet(img_student)
        out_s = out_s.view(test_bs, ncrops, -1).mean(1)   
        sloss = criterion(out_s, targets)
        PrivateTest_loss += sloss.item()
        sconf_mat, sacc, smAP, sF1_score = ACC_evaluation(sconf_mat, out_s, targets, NUM_CLASSES)
        
    return PrivateTest_loss/(batch_idx+1), 0.00, 0.00, 0.00, 100.*sacc, 100.*smAP, 100.*sF1_score

for epoch in range(0, total_epoch):
    if epoch == 0:
        train_loss, train_T_acc, train_T_mAP, train_T_F1, train_S_acc, train_S_mAP, train_S_F1 = train(epoch)
    else:
        pre_tnet = deepcopy(tnet).cuda().eval()
        valid(epoch)
        aft_tnet = deepcopy(tnet).cuda().eval()
        train_loss, train_T_acc, train_T_mAP, train_T_F1, train_S_acc, train_S_mAP, train_S_F1 = train(epoch)
    test_loss, test_T_acc, test_T_mAP, test_T_F1, test_S_acc, test_S_mAP, test_S_F1 = test(epoch)
    
    print("train_loss:  %0.3f, train_T_acc:  %0.3f, train_T_mAP:  %0.3f, train_T_F1:  %0.3f, train_S_acc:  %0.3f, train_S_mAP:  %0.3f, train_S_F1:  %0.3f"%(train_loss, train_T_acc, train_T_mAP, train_T_F1, train_S_acc, train_S_mAP, train_S_F1))
    print("test_loss:  %0.3f, test_T_acc:  %0.3f, test_T_mAP:  %0.3f, test_T_F1:  %0.3f, test_S_acc:  %0.3f, test_S_mAP:  %0.3f, test_S_F1:  %0.3f"%(test_loss, test_T_acc, test_T_mAP, test_T_F1, test_S_acc, test_S_mAP, test_S_F1))
    
    writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('epoch/Teacher_accuracy', {'train': train_T_acc, 'test': test_T_acc}, epoch)
    writer.add_scalars('epoch/Teacher_MAP', {'train': train_T_mAP, 'test': test_T_mAP}, epoch)
    writer.add_scalars('epoch/Teacher_F1', {'train': train_T_F1, 'test': test_T_F1}, epoch)
    writer.add_scalars('epoch/Student_accuracy', {'train': train_S_acc, 'test': test_S_acc}, epoch)
    writer.add_scalars('epoch/Student_MAP', {'train': train_S_mAP, 'test': test_S_mAP}, epoch)
    writer.add_scalars('epoch/Student_F1', {'train': train_S_F1, 'test': test_S_F1}, epoch)
    
    if test_T_acc > best_teacher_ACC:
        best_teacher_ACC= test_T_acc
        print ('Saving teacher models......')
        print("Test_Teacher_accuracy: %0.3f" % test_T_acc)
        print("Test_Teacher_MAP: %0.3f" % test_T_mAP)
        print("Test_Teacher_F1: %0.3f" % test_T_F1)
        state = {
            'Teacher': tnet.state_dict() if args.cuda else tnet,
            'test_Teacher_accuracy': test_T_acc,  
            'test_Teacher_MAP': test_T_mAP,  
            'test_Teacher_F1': test_T_F1, 
            'test_epoch': epoch,
        } 
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'Best_Teacher_model.t7'))
        
    if test_S_acc > best_student_ACC:
        best_student_ACC= test_S_acc
        best_student_mAP= test_S_mAP
        best_student_F1= test_S_F1
        print ('Saving student models......')
        print("Test_Student_accuracy: %0.3f" % test_S_acc)
        print("Test_Student_MAP: %0.3f" % test_S_mAP)
        print("Test_Student_F1: %0.3f" % test_S_F1)
        state = {
            'Student': snet.state_dict() if args.cuda else snet,
            'test_Student_accuracy': test_S_acc,  
            'test_Student_MAP': test_S_mAP,  
            'test_Student_F1': test_S_F1, 
            'test_epoch': epoch,
        } 
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'Best_Student_model.t7'))
        
print("best_PrivateTest_acc: %0.2f" % best_student_ACC)
print("best_PrivateTest_mAP: %0.2f" % best_student_mAP)
print("best_PrivateTest_F1: %0.2f" % best_student_F1)        
writer.close()
