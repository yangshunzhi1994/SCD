import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils.utils_metrics import evaluate
from losses import KL_divergence_sample, KL_divergence, Fuzzy_PID


def Train(t_model, s_model, pre_tnet, aft_tnet, t_optimizer, s_optimizer, epoch, gen):
    TP, T = 0.06, 1
    total_CE_loss = 0
    total_accuracy = 0
    t_model.train()
    s_model.train()
    train_loss = 0
    for iteration, (images, labels, index) in enumerate(gen):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

        t_optimizer.zero_grad()
        _, _, _, t_outputs2 = t_model(images, "train")
        t_loss = nn.NLLLoss(reduction='none')(F.log_softmax(t_outputs2, dim = -1), labels)
        if epoch == 0:
            t_loss = t_loss.mean()
        else:
            with torch.no_grad():
                _, _, _, pre_outputs2 = pre_tnet(images, "train")
                _, _, _, aft_outputs2 = aft_tnet(images, "train")
            t_pre_loss = nn.NLLLoss(reduction='none')(F.log_softmax(pre_outputs2, dim=-1), labels)
            t_aft_loss = nn.NLLLoss(reduction='none')(F.log_softmax(aft_outputs2, dim=-1), labels)
            P_loss = t_pre_loss - t_aft_loss
            u = TP * P_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            t_loss = torch.dot(t_loss, weights)
        t_loss.backward()
        t_optimizer.step()

        s_optimizer.zero_grad()
        _, _, _, s_outputs2 = s_model(images, "train")
        s_CE_loss = nn.NLLLoss()(F.log_softmax(s_outputs2, dim=-1), labels)
        with torch.no_grad():
            _, _, _, t_outputs2 = t_model(images, "train")

        if epoch == 0:
            sloss = KL_divergence(temperature=T).cuda()(t_outputs2, s_outputs2) + s_CE_loss
        else:
            sloss = KL_divergence_sample(temperature=T).cuda()(t_outputs2, s_outputs2, weights.detach()) + s_CE_loss
        sloss.backward()
        s_optimizer.step()
        train_loss += sloss.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(t_outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
        total_CE_loss += s_CE_loss.item()
        total_accuracy += accuracy.item()

    return total_CE_loss / (iteration + 1), total_accuracy / (iteration + 1)

def set_PID(P, I, D):
    SP, SI, SD = P*3, I, D
    return SP, SI, SD

def Valid(t_model, s_model, t_optimizer, epoch, gen, last_loss, lastlast_loss, meta_indices):
    P, I, D, L_m, M = 0.04, 0.0, 0.1, 2.1, 7
    meta = deepcopy(s_model)
    meta.eval()
    t_model.train()
    SP, SI, SD = set_PID(P, I, D)

    if epoch > 1:
        err = last_loss.mean()
        SP, SI, SD = Fuzzy_PID(SP, SI, SD, err, L_m, P, M)

    for iteration, batch in enumerate(gen):
        images, labels, mask = batch
        index = torch.zeros_like(mask).cuda()
        for i in range(len(index)):
            index[i] = np.squeeze(np.squeeze(np.argwhere(meta_indices == mask[i]), 0), 0)

        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
            _, _, _, s_outputs2 = s_model(images, "train")
            s_CE_loss = nn.NLLLoss(reduction='none')(F.log_softmax(s_outputs2, dim=-1), labels)
        t_optimizer.zero_grad()
        _, _, _, t_outputs2 = t_model(images, "train")
        t_loss = nn.NLLLoss(reduction='none')(F.log_softmax(t_outputs2, dim=-1), labels)

        if epoch == 1:
            u = SP * s_CE_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            t_loss = torch.dot(t_loss, weights)
            t_loss.backward()
            t_optimizer.step()
            last_loss[index] = s_CE_loss

        elif epoch == 2:
            k = torch.div(s_CE_loss + last_loss[index], 2)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k

            D_loss = s_CE_loss - last_loss[index]

            u = SP * s_CE_loss + SI * I_loss + SD * D_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            t_loss = torch.dot(t_loss, weights)
            t_loss.backward()
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = s_CE_loss
        else:
            k = torch.div(s_CE_loss + last_loss[index] + lastlast_loss[index], 3)
            I_loss = torch.div(2, max(k) - min(k)) * (k - min(k)) - 1
            I_loss = torch.tanh(I_loss) * k

            D_loss = s_CE_loss - 2 * last_loss[index] + lastlast_loss[index]

            u = SP * s_CE_loss + SI * I_loss + SD * D_loss
            weights = F.softmax(u, dim=0)
            weights.requires_grad = True
            t_loss = torch.dot(t_loss, weights)
            t_loss.backward()
            t_optimizer.step()
            lastlast_loss[index] = last_loss[index]
            last_loss[index] = s_CE_loss
        return SP, SI, SD, last_loss, lastlast_loss


def Test(model, test_loader):
    model.eval()
    labels, distances = [], []
    for _, (data_a, data_p, label) in enumerate(test_loader):
        with torch.no_grad():
            data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
            data_a, data_p = data_a.cuda(), data_p.cuda()
            out_a, out_p = model(data_a), model(data_p)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
        distances.append(dists.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])
    tpr, fpr, accuracy, val, val_std, far, best_thresholds = evaluate(distances, labels)
    return accuracy, val, val_std, far, best_thresholds





def Train_teacher(t_model, t_optimizer, gen):
    total_CE_loss = 0
    total_accuracy = 0
    t_model.train()
    for iteration, (images, labels, index) in enumerate(gen):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
        t_optimizer.zero_grad()
        _, _, _, t_outputs2 = t_model(images, "train")
        t_loss = nn.NLLLoss()(F.log_softmax(t_outputs2, dim=-1), labels)
        t_loss.backward()
        t_optimizer.step()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(t_outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
        total_CE_loss += t_loss.item()
        total_accuracy += accuracy.item()
        
    return total_CE_loss / (iteration + 1), total_accuracy / (iteration + 1)