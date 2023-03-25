# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import skfuzzy as fuzz
from Fuzzy_PID import fuzzyP3, fuzzyP5, fuzzyP7, fuzzyP9, fuzzyP11, fuzzyP13, fuzzyP15, fuzzyP17


def Fuzzy_PID(scale_P, scale_I, scale_D, err, max_err, kd_P, M):

    if M==3:
        fuzzy_p_err = fuzzyP3(err.cpu(), max_err) # fuzzy_p_err = fuzzyP3(err.cpu(), max_err)
    elif M==5:
        fuzzy_p_err = fuzzyP5(err.cpu(), max_err)
    elif M==7:
        fuzzy_p_err = fuzzyP7(err.cpu(), max_err)
    elif M==9:
        fuzzy_p_err = fuzzyP9(err.cpu(), max_err)
    elif M==11:
        fuzzy_p_err = fuzzyP11(err.cpu(), max_err)
    elif M==13:
        fuzzy_p_err = fuzzyP13(err.cpu(), max_err)
    elif M==15:
        fuzzy_p_err = fuzzyP15(err.cpu(), max_err)
    elif M==17:
        fuzzy_p_err = fuzzyP17(err.cpu(), max_err)
    else:
        raise Exception('Invalid M...')

    P = np.arange(0.0, M * kd_P - 0.00001, kd_P)
    change_P = fuzz.defuzz(P, np.asarray(fuzzy_p_err), 'centroid')
    scale_I = scale_I + (scale_P - change_P)
    scale_D = scale_D + (scale_P - change_P)

    scale_P = torch.from_numpy(np.array(change_P)).cuda().float()
    scale_I = torch.clamp(torch.from_numpy(np.array(scale_I)), 0.0, M * kd_P).cuda().squeeze().float()
    scale_D = torch.clamp(torch.from_numpy(np.array(scale_D)), 0.0, M * kd_P).cuda().squeeze().float()

    return scale_P, scale_I, scale_D



    
class KL_divergence(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit):
        KD_loss = nn.KLDivLoss()(F.log_softmax(student_logit/self.T + 1e-8, dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T
        return KD_loss
    
class KL_divergence_sample(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_divergence_sample, self).__init__()
        self.T = temperature
    def forward(self, teacher_logit, student_logit, weights):
        KD_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(student_logit/self.T + 1e-8, dim=1), F.softmax(teacher_logit/self.T,dim=1)) * self.T * self.T
        KD_loss = torch.dot(KD_loss.sum(1), weights)
        return KD_loss



