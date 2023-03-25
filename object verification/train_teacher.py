import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse
import numpy as np
from nets.facenet import Facenet
from nets.facenet_training import set_optimizer
from utils.dataloader import FacenetDataset, LFWDataset
from utils.utils_fit import Train_teacher, Test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
    parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
    parser.add_argument('--model', type=str, default="Teacher", help='Teacher')
    parser.add_argument('--data_name', type=str, default="IJBC", help='IJBC')
    parser.add_argument('--test_data', type=str, default="IJBC", help='IJBC')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--train_bs', default=32, type=int, help='Batch size')# 96
    parser.add_argument('--test_bs', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--preload', type=str, default="False", help='Whether to preload')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    best_accuracy = 0
    path = os.path.join(args.save_root + args.data_name + '_' + args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    text_path = path + '/' + args.data_name + '_Teacher.txt'
    num_classes = 22000
    input_shape = [112, 112, 3]
    t_model = Facenet(backbone="Teacher", num_classes=num_classes).cuda()
    LFW_loader = torch.utils.data.DataLoader(LFWDataset(image_size=input_shape, data_name=args.test_data), batch_size=args.test_bs, shuffle=False)
    if True:
        t_optimizer = optim.SGD(t_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_dataset = FacenetDataset(input_shape)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=args.train_bs, num_workers=args.num_workers, pin_memory=True)

        for epoch in range(0, args.epochs):
            steps = np.sum(epoch > np.asarray([30, 60, 90]))
            if steps > 0:
                lr = args.lr * (0.1 ** steps)
                set_optimizer(t_optimizer, lr)
            else:
                lr = args.lr
            time1 = time.time()
            train_CE_loss, train_accuracy = Train_teacher(t_model, t_optimizer, gen)
            time2 = time.time()
            test_accuracy, val, val_std, far, best_thresholds = Test(t_model, LFW_loader)
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
                    'tnet': t_model.state_dict(),
                    'best_PrivateTest_acc': test_accuracy,
                    'Best_thresholds': best_thresholds,
                    'val': val,
                    'val_std': val_std,
                    'far': far,
                }
                torch.save(state, os.path.join(path, 'Best_Teacher_model.t7'))

            f = open(text_path, 'a')
            f.write("\n")
            f.close()