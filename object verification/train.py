import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
from copy import deepcopy
from nets.facenet import Facenet
from nets.facenet_training import set_optimizer
from utils.dataloader import FacenetDataset, LFWDataset
from utils.utils_fit import Train, Valid, Test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
    parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
    parser.add_argument('--model', type=str, default="MetaStudent", help='MetaStudent')
    parser.add_argument('--data_name', type=str, default="IJBC", help='IJBC')
    parser.add_argument('--test_data', type=str, default="IJBC", help='IJBC')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--train_bs', default=128, type=int, help='Batch size')# 96
    parser.add_argument('--test_bs', default=256, type=int, help='Batch size')
    parser.add_argument('--P', default=0.04, type=float, help='P')
    parser.add_argument('--I', default=0.0, type=float, help='I')
    parser.add_argument('--D', default=0.1, type=float, help='D')
    parser.add_argument('--TP', default=0.06, type=float, help='TP')
    parser.add_argument('--T', default=1, type=int, help='T')
    parser.add_argument('--L_m', default=2.1, type=float, help='L_m')
    parser.add_argument('--M', default=7, type=int, help='M')
    parser.add_argument('--Vali_num', default=5000, type=int, help='validation set')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--preload', type=str, default="False", help='Whether to preload')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    input_shape = [112, 112, 3]
    best_accuracy = 0
    path = args.save_root + args.data_name + '_' + args.model + '_P_' + str(args.P) + '_I_' + str(args.I) + '_D_' \
           + str(args.D) + '_TP_' + str(args.TP) + str(args.T) + '_T_' + '_Lm_' + str(args.L_m) + '_val_' + str(args.Vali_num) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    text_path = path + args.data_name + '_' + args.model + '.txt'
    num_classes = 22000
    t_model = Facenet(backbone="Teacher", num_classes=num_classes).cuda()
    s_model = Facenet(backbone="Student", num_classes=num_classes).cuda()
    LFW_loader = torch.utils.data.DataLoader(LFWDataset(image_size=input_shape, data_name=args.test_data), batch_size=args.test_bs, shuffle=False)

    if True:
        t_optimizer = optim.SGD(t_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        s_optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_dataset = FacenetDataset(input_shape)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - args.Vali_num, args.Vali_num],
                                                           generator=torch.Generator().manual_seed(0))
        n_data = len(val_dataset)
        meta_indices = val_dataset.indices
        list.sort(meta_indices)
        meta_indices = torch.as_tensor(meta_indices)
        last_loss = torch.zeros(n_data).cuda()
        lastlast_loss = torch.zeros(n_data).cuda()

        gen = DataLoader(train_dataset, shuffle=True, batch_size=args.train_bs, num_workers=args.num_workers, pin_memory=True)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.train_bs, num_workers=args.num_workers, pin_memory=True)

        for epoch in range(0, args.epochs):
            steps = np.sum(epoch > np.asarray([30, 60, 90]))
            if steps > 0:
                lr = args.lr * (0.1 ** steps)
                set_optimizer(t_optimizer, lr)
                set_optimizer(s_optimizer, lr)
            else:
                lr = args.lr
            time1 = time.time()
            if epoch == 0:
                train_CE_loss, train_accuracy = Train(t_model, s_model, None, None, t_optimizer, s_optimizer, epoch, gen)
            else:
                pre_tnet = deepcopy(t_model).cuda().eval()
                SP, SI, SD, last_loss, lastlast_loss = Valid(t_model, s_model, t_optimizer, epoch, gen_val, last_loss, lastlast_loss, meta_indices)
                loss_state = {
                    'last_loss': last_loss,
                    'SP': SP,
                    'SI': SI,
                    'SD': SD,
                    'last_loss_mean': last_loss.mean(),
                }
                torch.save(loss_state, path + '/last_loss_' + str(epoch) + '.pth')
                f = open(text_path, 'a')
                f.write("\nEpoch: %d, SP:  %0.2f, SI:  %0.2f, SD:  %0.2f" % (epoch, SP, SI, SD))
                f.close()
                aft_tnet = deepcopy(t_model).cuda().eval()
                train_CE_loss, train_accuracy = Train(t_model, s_model, pre_tnet, aft_tnet, t_optimizer, s_optimizer, epoch, gen)
            time2 = time.time()
            test_accuracy, val, val_std, far, best_thresholds = Test(s_model, LFW_loader)
            f = open(text_path, 'a')
            f.write("\nEpoch: %d, train_CE_loss:  %0.3f, train_accuracy:  %0.2f, total time:  %0.3f, learning_rate:  %0.6f" %
                    (epoch, train_CE_loss, 100.*train_accuracy, time2 - time1, lr))
            f.write("\nEpoch: %d, Test_accuracy: %2.5f+-%2.5f, Validation rate: %2.5f+-%2.5f @ FAR=%2.5f, Best_thresholds: %2.5f\n"
                    % (epoch, np.mean(test_accuracy), np.std(test_accuracy), val, val_std, far, best_thresholds))
            f.close()

            if test_accuracy.mean() > best_accuracy:
                best_accuracy = test_accuracy.mean()
                f = open(text_path, 'a')
                f.write('\nSaving models......')
                f.write('\nAccuracy: %2.5f+-%2.5f' % (np.mean(test_accuracy), np.std(test_accuracy)))
                f.write('\nBest_thresholds: %2.5f' % best_thresholds)
                f.write('\nValidation rate: %2.5f+-%2.5f @ FAR=%2.5f\n' % (val, val_std, far))
                f.close()

                state = {
                    'snet': s_model.state_dict(),
                    'best_PrivateTest_acc': test_accuracy,
                    'Best_thresholds': best_thresholds,
                    'val': val,
                    'val_std': val_std,
                    'far': far,
                }
                torch.save(state, os.path.join(path, 'Best_Student_model.t7'))

            f = open(text_path, 'a')
            f.write("\n")
            f.close()