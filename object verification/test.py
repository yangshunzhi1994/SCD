import os
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from nets.facenet import Facenet
from utils.utils import load_pretrained_model_Filter
from utils.dataloader import LFWDataset
from utils.utils_fit import Test

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--test_data', type=str, default="IJBC", help='IJBC, LFW, SCface, Tinyface')
parser.add_argument('--test_bs', default=64, type=int, help='Batch size')
parser.add_argument('--model', type=str, default="CRD", help='ProKT, CRD, FKD, Jiang, SCKD, Annealing-KD, Robust, '
                                  'KDEP, RAD, SKD, Massoli, DGKD,Filter-KD,HuangEKD,DKD,EEM,CRKD')
args = parser.parse_args()
num_classes = 22000
input_shape = [112, 112, 3]
LFW_loader = torch.utils.data.DataLoader(LFWDataset(image_size=input_shape, data_name=args.test_data), batch_size=args.test_bs, shuffle=False)
path = os.path.join('results/IJBC_' + args.model + '/Best_Student_model.t7')
s_model = Facenet(backbone="Student", num_classes=num_classes).cuda()
scheckpoint = torch.load(path)
load_pretrained_model_Filter(s_model, scheckpoint['snet'])
test_accuracy, val, val_std, far, best_thresholds = Test(s_model, LFW_loader)

text_path = 'results/Test_' + args.test_data + '.txt'
f = open(text_path, 'a')
f.write("\nModel: %s, Test_accuracy: %2.5f+-%2.5f, Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, Best_thresholds: %2.5f\n"
        % (args.model, np.mean(test_accuracy), np.std(test_accuracy), val, val_std, far, best_thresholds))
f.close()