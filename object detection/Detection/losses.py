# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import skfuzzy as fuzz

    
class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit):
        KD_loss = nn.KLDivLoss()(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T
        return KD_loss
    
class KL_divergence_sample(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence_sample, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit, weights):
        KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(student_logit/self.T,dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T
        KD_loss = torch.dot(KD_loss.sum(1), weights)
        return KD_loss


# membership function
def fuzzyP(x, max_err):
    # input must be rescaled
    # output is a list of membership
    # [NB, NM, NS, ZE, PS, PM, PB]
    membership = [0, 0, 0, 0, 0, 0, 0]
    bisection_error = 1 / 7 * max_err
    # NB
    if x <= bisection_error:
        membership[0] = 1
    elif bisection_error < x and x <= 2 * bisection_error:
        membership[0] = (2 * bisection_error - x) / bisection_error
    # NM
    if bisection_error < x and x <= 2 * bisection_error:
        membership[1] = (x - bisection_error) / bisection_error
    elif 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[1] = (3 * bisection_error - x) / bisection_error
    # NS
    if 2 * bisection_error < x and x <= 3 * bisection_error:
        membership[2] = (x - 2 * bisection_error) / bisection_error
    elif 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[2] = (4 * bisection_error - x) / bisection_error
    # ZE
    if 3 * bisection_error < x and x <= 4 * bisection_error:
        membership[3] = (x - 3 * bisection_error) / bisection_error
    elif 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[3] = (5 * bisection_error - x) / bisection_error
    # PS
    if 4 * bisection_error < x and x <= 5 * bisection_error:
        membership[4] = (x - 4 * bisection_error) / bisection_error
    elif 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[4] = (6 * bisection_error - x) / bisection_error
    # PM
    if 5 * bisection_error < x and x <= 6 * bisection_error:
        membership[5] = (x - 5 * bisection_error) / bisection_error
    elif 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[5] = (7 * bisection_error - x) / bisection_error
    # PB
    if 6 * bisection_error < x and x <= 7 * bisection_error:
        membership[6] = (x - 6 * bisection_error) / bisection_error
    elif 7 * bisection_error <= x:
        membership[6] = 1
    return membership


def Fuzzy_PID(scale_P, scale_I, scale_D, err, max_err, text_path, kd_P, epoch):
    fuzzy_p_err = fuzzyP(err.cpu(), max_err)  # get membership vector
    P = np.arange(0.0, 7 * kd_P - 0.00001, kd_P)
    change_P = fuzz.defuzz(P, np.asarray(fuzzy_p_err), 'centroid')

    scale_I = scale_I + (scale_P - change_P)
    scale_D = scale_D + (scale_P - change_P)

    scale_P = torch.from_numpy(np.array(change_P)).cuda().float()
    scale_I = torch.clamp(torch.from_numpy(np.array(scale_I)), 0.0, 6 * kd_P).cuda().squeeze().float()
    scale_D = torch.clamp(torch.from_numpy(np.array(scale_D)), 0.0, 6 * kd_P).cuda().squeeze().float()
    f = open(text_path, 'a')
    f.write("\nEpoch: %d, KP:  %0.3f, KI:  %0.3f, KD:  %0.3f" % (epoch, scale_P, scale_I, scale_D))
    f.close()

    return scale_P, scale_I, scale_D