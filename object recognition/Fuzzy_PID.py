# coding: utf-8
import numpy as np
import torch
import skfuzzy as fuzz

# membership function
def fuzzyP(x):
    # input must be rescaled
    # output is a list of membership
    # [NB, NM, NS, ZE, PS, PM, PB]
    membership = [0, 0, 0, 0, 0, 0, 0]
    # NB
    if x <= 0.3:
        membership[0] = 1
    elif 0.3 < x and x <= 0.6:
        membership[0] = (0.6 - x) / 0.3
    # NM
    if 0.3 < x and x <= 0.6:
        membership[1] = (x - 0.3) / 0.3
    elif 0.6 < x and x<= 0.9:
        membership[1] = (0.9 - x) / 0.3
    # NS
    if 0.6 < x and x<= 0.9:
        membership[2] = (x - 0.6) / 0.3
    elif 0.9 < x and x <= 1.2:
        membership[2] = (1.2 - x) / 0.3
    # ZE
    if 0.9 < x and x <= 1.2:
        membership[3] = (x - 0.9) / 0.3
    elif 1.2 < x and x <= 1.5:
        membership[3] = (1.5 - x) / 0.3
    # PS
    if 1.2 < x and x <= 1.5:
        membership[4] = (x - 1.2) / 0.3
    elif 1.5 < x and x <= 1.8:
        membership[4] = (1.8 - x) / 0.3
    # PM
    if 1.5 < x and x <= 1.8:
        membership[5] = (x - 1.5) / 0.3
    elif 1.8 < x and x <= 2.1:
        membership[5] = (2.1 - x) / 0.3
    # PB
    if 1.8 < x and x <= 2.1:
        membership[6] = (x - 1.8) / 0.3
    elif 2.1 <= x:
        membership[6] = 1
    return membership

def Fuzzy_PID(scale_P, scale_I, scale_D, err):
    
    fuzzy_p_err = fuzzyP(err.cpu()) # get membership vector
    P=np.arange(0.0,0.7,0.1)
    change_P = fuzz.defuzz(P, np.asarray(fuzzy_p_err), 'centroid')
    
    scale_I = scale_I + (scale_P - change_P)
    scale_D = scale_D + (scale_P - change_P)
    
    scale_P = torch.from_numpy(np.array(change_P)).cuda().float()
    scale_I = torch.clamp(torch.from_numpy(np.array(scale_I)), 0.0, 0.6).cuda().squeeze().float()
    scale_D = torch.clamp(torch.from_numpy(np.array(scale_D)), 0.0, 0.6).cuda().squeeze().float()
    
    print (scale_P)
    print (scale_I)
    print (scale_D)
    return scale_P, scale_I, scale_D