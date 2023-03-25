# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F





    
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



