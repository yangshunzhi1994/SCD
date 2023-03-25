from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch

def load_pretrained_model(net, resume_net):
    print('Loading resume network...')
    state_dict = torch.load(resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def load_pretrained_model_Filter(net, state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr