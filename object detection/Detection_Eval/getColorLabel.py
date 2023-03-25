#!/usr/bin/env python
import pylab
import numpy as np
import os
colorCounter = 11
color_list = pylab.cm.Set1(np.linspace(0, 1, 200))


def getColorLabel(name):
    print (name)
    global colorCounter, color_list
    if name.find("RAD") != -1:
        color = 'C0'
        label = "Ge et al. [17]"
    elif name.find("SKD") != -1:
        color = 'C1'
        label = "Ge et al. [18]"
    elif name.find("EEM") != -1:
        color = 'slateblue'
        label = "EEM [10]"
    elif name.find("CRKD") != -1:
        color = 'C3'
        label = "CRKD [43]"
    elif name.find("Robust") != -1:
        color = 'C4'
        label = "Robust [11]"
    elif name.find("CRD") != -1:
        color = 'C5'
        label = "CRD [12]"
    elif name.find("DKD") != -1:
        color = 'C6'
        label = "DKD [13]"
    elif name.find("Filter-KD") != -1:
        color = 'C7'
        label = "Filter-KD [14]"
    elif name.find("Jiang") != -1:
        color = 'C8'
        label = "Jiang et al. [15]"
    elif name.find("FKD") != -1:
        color = 'C9'
        label = "FKD [16]"
    elif name.find("Massoli") != -1:
        color = 'aquamarine'
        label = "Massoli et al. [19]"
    elif name.find("KDEP") != -1:
        color = 'blue'
        label = "KDEP [20]"
    elif name.find("Huang") != -1:
        color = 'burlywood'
        label = "Huang et al. [22]"
    elif name.find("Annealing") != -1:
        color = 'cadetblue'
        label = "Annealing-KD [24]"
    elif name.find("SCKD") != -1:
        color = 'chocolate'
        label = "SCKD [26]"
    elif name.find("DGKD") != -1:
        color = 'cornflowerblue'
        label = "DGKD [27]"
    elif name.find("MetaDistil") != -1:
        color = 'mediumaquamarine'
        label = "MetaDistil [31]"
    elif name.find("MPL") != -1:
        color = 'darkkhaki'
        label = "MPL [29]"
    elif name.find("IKD") != -1:
        color = 'darkolivegreen'
        label = "IKD [30]"
    elif name.find("Leap") != -1:
        color = 'navy'
        label = "Leap [21]]"
    elif name.find("Lazy") != -1:
        color = 'slategray'
        label = "\"Lazy\" MAML [23]"
    elif name.find("ProKT") != -1:
        color = 'tomato'
        label = "ProKT [25]"
    elif name.find("SCD") != -1:
        color = 'C22'
        label = "SCD"
    else:
        color = 'C3'
        label = "XXXX"

    return [color, label]
