from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import torch.nn.functional as F
from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss, MultiBoxLoss_valid
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
import numpy as np
import torch.nn as nn
from copy import deepcopy
from models.faceboxes import FaceBoxes
from models.faceboxes_student import FaceBoxes_S
from losses import KL_divergence_sample, KL_divergence, Fuzzy_PID
from utils.utils import set_lr

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--training_dataset', default='../datasets/Detection/WIDER_FACE', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--ngpu', default=0, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--model', type=str, default="MetaStudent", help='MetaStudent')
parser.add_argument('--P', default=0.04, type=float, help='P')
parser.add_argument('--I', default=0.0, type=float, help='I')
parser.add_argument('--D', default=0.1, type=float, help='D')
parser.add_argument('--TP', default=0.06, type=float, help='TP')
parser.add_argument('--T', default=20, type=int, help='T')
parser.add_argument('--L_m', default=2.1, type=float, help='L_m')
parser.add_argument('--Vali_num', default=1000, type=int, help='Total number of validation sets')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

path = args.save_folder + args.model + '_P_' + str(args.P) + '_I_' + str(args.I) + '_D_' + str(args.D) \
            + '_TP_' + str(args.TP) + '_T_' + str(args.T) + '_Lm_' + str(args.L_m) + '_Vali_' + str(args.Vali_num) + '/'
if not os.path.exists(path):
    os.makedirs(path)
text_path = path + 'Ours.txt'
f = open(text_path, 'a')
f.write('\nThe training mode is:                            '+ str(args.model))
f.write('\nThe P is:                                        '+ str(args.P))
f.write('\nThe I is:                                        '+ str(args.I))
f.write('\nThe D is:                                        '+ str(args.D))
f.write('\nThe T is:                                        '+ str(args.T))
f.write('\nThe TP is:                                       '+ str(args.TP))
f.write('\nThe Lm is:                                       '+ str(args.L_m))
f.close()

img_dim = 1024 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
num_gpu = args.ngpu
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_dataset = args.training_dataset
save_folder = args.save_folder
gpu_train = cfg['gpu_train']

t_net = FaceBoxes('train', img_dim, num_classes)
s_net = FaceBoxes_S('train', img_dim, num_classes)

if args.resume_net is not None:
    print('Start loading teacher network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    t_net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    t_net = torch.nn.DataParallel(t_net, device_ids=list(range(num_gpu)))

device = torch.device('cuda:0' if gpu_train else 'cpu')
cudnn.benchmark = True
t_net = t_net.to(device)
s_net = s_net.to(device)

t_optimizer = optim.SGD(t_net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
s_optimizer = optim.SGD(s_net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
criterion_valid = MultiBoxLoss_valid(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)

dataset = VOCDetection(training_dataset, preproc(img_dim, rgb_mean), AnnotationTransform())
trainset, meta_set = torch.utils.data.random_split(dataset, [12876 - args.Vali_num, args.Vali_num], generator=torch.Generator().manual_seed(0))
n_data = len(meta_set)
meta_indices = meta_set.indices
list.sort(meta_indices)
meta_indices = torch.as_tensor(meta_indices)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
validloader = torch.utils.data.DataLoader(meta_set, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
last_loss = torch.zeros(n_data).cuda()
lastlast_loss = torch.zeros(n_data).cuda()

pre_tnet, aft_tnet = None, None
def set_PID():
    SP, SI, SD = args.P*3, args.I, args.D
    return SP, SI, SD

def valid(epoch):
    meta = deepcopy(s_net)
    meta.eval()
    t_net.train()
    SP, SI, SD = set_PID()

    if epoch > 1:
        err = last_loss.mean()
        SP, SI, SD = Fuzzy_PID(SP, SI, SD, err, args.L_m, text_path, args.P, epoch)

    for batch, (images, targets, mask) in enumerate(validloader):
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]
        index = torch.zeros_like(mask).cuda()
        for i in range(len(index)):
            index[i] = np.squeeze(np.squeeze(np.argwhere(meta_indices == mask[i]), 0), 0)
        _, _, t_out = t_net(images)
        t_out, t_targets = criterion_valid(t_out, priors, targets)
        t_optimizer.zero_grad()
        with torch.no_grad():
            _, _, m_out = meta(images)
            m_out, m_targets = criterion_valid(m_out, priors, targets)

        sloss = torch.FloatTensor(len(images)).cuda()
        tloss = torch.FloatTensor(len(images)).cuda()
        for i in range(0, len(images)):
            tloss[i] = F.cross_entropy(t_out[i], t_targets[i])
            sloss[i] = F.cross_entropy(m_out[i], m_targets[i])

        if epoch == 1:
            u = SP * sloss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            t_optimizer.step()
            last_loss[index] = sloss

        elif epoch == 2:
            k = torch.div(sloss + last_loss[index], 2)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k

            D_loss = sloss - last_loss[index]

            u = SP * sloss + SI * I_loss + SD * D_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = sloss
        else:
            k = torch.div(sloss + last_loss[index] + lastlast_loss[index], 3)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k

            D_loss = sloss - 2 * last_loss[index] + lastlast_loss[index]

            u = SP * sloss + SI * I_loss + SD * D_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            tloss = torch.dot(tloss, weights)
            tloss.backward()
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = sloss
    return SP, SI, SD

def train(epoch):
    global step_index, pre_tnet, aft_tnet
    t_net.train()
    s_net.train()
    train_loss = 0
    steps = np.sum(epoch > np.asarray([200, 250]))
    if steps > 0:
        lr = args.lr * (0.1 ** steps)
        set_lr(t_optimizer, lr)
        set_lr(s_optimizer, lr)
    else:
        lr = args.lr

    for batch_idx, (images, targets, index) in enumerate(trainloader):
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]
        _, _, t_out = t_net(images)
        t_optimizer.zero_grad()
        t_loss_l, t_out, t_targets = criterion(t_out, priors, targets)
        tloss = F.cross_entropy(t_out, t_targets, reduction='none')
        if epoch == 0:
            tloss = tloss.mean()
        else:
            with torch.no_grad():
                _, _, pre_out = pre_tnet(images)
                _, _, aft_out = aft_tnet(images)
                _, pre_out, pre_targets = criterion(pre_out, priors, targets)
                _, aft_out, aft_targets = criterion(aft_out, priors, targets)
            pre_tloss = F.cross_entropy(pre_out, pre_targets, reduction='none')
            aft_tloss = F.cross_entropy(aft_out, aft_targets, reduction='none')
            P_loss = pre_tloss - aft_tloss
            u = args.TP * P_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            tloss = torch.dot(tloss, weights)
        tloss = tloss + cfg['loc_weight'] * t_loss_l
        tloss.backward()
        t_optimizer.step()

        s_optimizer.zero_grad()
        _, _, s_out = s_net(images)
        s_loss_l, s_out, s_targets = criterion(s_out, priors, targets)
        with torch.no_grad():
            _, _, t_out = t_net(images)
            _, t_out, _ = criterion(t_out, priors, targets)
        if epoch == 0:
            sloss = KL_divergence(temperature=args.T).cuda()(t_out, s_out) + cfg['loc_weight'] * s_loss_l + F.cross_entropy(s_out, s_targets)
        else:
            sloss = KL_divergence_sample(temperature=args.T).cuda()(t_out, s_out, weights.detach()) + cfg['loc_weight'] * s_loss_l + F.cross_entropy(s_out, s_targets)
        sloss.backward()
        s_optimizer.step()
        train_loss += sloss.item()
    return train_loss / (batch_idx + 1), lr

if __name__ == '__main__':
    for epoch in range(0, max_epoch):
        time1 = time.time()
        if epoch == 0:
            train_loss, lr = train(epoch)
        else:
            pre_tnet = deepcopy(t_net).cuda().eval()
            SP, SI, SD = valid(epoch)
            loss_state = {
                'last_loss': last_loss,
                'SP': SP,
                'SI': SI,
                'SD': SD,
                'last_loss_mean': last_loss.mean(),
            }
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(loss_state, path + '/last_loss_' + str(epoch) + '.pth')
            aft_tnet = deepcopy(t_net).cuda().eval()
            train_loss, lr = train(epoch)
        time2 = time.time()

        f = open(text_path, 'a')
        f.write("\nEpoch: %d, train_loss:  %0.3f, last_loss_mean:  %0.3f, total time:  %0.3f, learning_rate:  %0.6f" %
                (epoch, train_loss, last_loss.mean(), time2 - time1, lr))
        f.close()

        if (epoch % 30 == 0):
            torch.save(s_net.state_dict(), path + 'Ours_' + str(epoch) + '.pth')

torch.save(s_net.state_dict(), path + 'Ours' + '.pth')
torch.save(t_net.state_dict(), path + 'teacher_Ours' + '.pth')