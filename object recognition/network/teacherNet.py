import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

import torch.utils.checkpoint as cp

def _bn_function_factory(conv, norm, relu):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = relu(norm(conv(concated_features)))
        return bottleneck_output

    return bn_function
    
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('conv0', nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=3, padding=1)),
        self.add_module('norm0', nn.BatchNorm2d(4 * growth_rate)),
        self.add_module('relu0', nn.ReLU(inplace=True)),
        
        self.add_module('conv1', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)),
        self.add_module('norm1', nn.BatchNorm2d(growth_rate)),

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.conv0, self.norm0, self.relu0)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.norm1(self.conv1(bottleneck_output))
        return new_features

class EdgeBlock(nn.Module):
    def __init__(self, nChannels, growth_rate):
        super(EdgeBlock, self).__init__()
        self.layer = _DenseLayer(nChannels, growth_rate)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = self.layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class resblock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(resblock, self).__init__()	

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)	
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu1  = nn.ReLU()

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)	
		self.bn2   = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		residual = x

		x = self.relu1(self.bn1(self.conv1(x))) 

		x = self.bn2(self.conv2(x)) 

		pout = residual + x

		return pout

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.pool(x)
        return out


class Teacher(nn.Module):
    def __init__(self, ResNet_factor=4, num_classes=7):
        super(Teacher, self).__init__()

        nChannels = 32
        growthRate = 16
        nDenseBlocks = [4, 4, 7]
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks[0])
        nChannels += nDenseBlocks[0]*growthRate
        self.res2 = self.make_layer(resblock, ResNet_factor, nChannels, nChannels)
        self.trans1 = Transition(nChannels, nChannels)

        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks[1])
        nChannels += nDenseBlocks[1]*growthRate
        self.res3 = self.make_layer(resblock, ResNet_factor, nChannels, nChannels)
        self.trans2 = Transition(nChannels, nChannels)

        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks[2])
        nChannels += nDenseBlocks[2]*growthRate
        self.res4 = self.make_layer(resblock, ResNet_factor, nChannels, nChannels)

        self.fc = nn.Linear(nChannels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(EdgeBlock(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def make_layer(self, block, num, in_channels, out_channels):  # num must >=2
        layers = [block(in_channels, out_channels)]
        for i in range(num-2):
            layers.append(block(out_channels, out_channels))
        layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        rb1 = self.dense1(out)
        rb1 = self.res2(rb1)

        rb2 = self.dense2(self.trans1(rb1))
        rb2 = self.res3(rb2)
        
        rb3 = self.dense3(self.trans2(rb2))
        rb3 = self.res4(rb3)
        
        feat = F.relu(rb3, inplace=True)
        mimic = F.avg_pool2d(feat, kernel_size=11).view(feat.size(0), -1)
        out = self.fc(mimic)
        return rb1, rb2, rb3, feat, mimic, out


