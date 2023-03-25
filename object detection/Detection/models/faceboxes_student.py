import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x


class FaceBoxes_S(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes_S, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size

    self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
    self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)

    self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

    self.loc, self.conf = self.multibox(self.num_classes)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

  def forward(self, x): # (32, 3, 1024, 1024)

    detection_sources = list()
    loc = list()
    conf = list()
    x = self.conv1(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    detection_sources.append(x) # (32, 128, 32, 32)

    x = self.conv3_1(x)
    x = self.conv3_2(x)
    detection_sources.append(x)  # (32, 256, 16, 16)

    x = self.conv4_1(x)
    x = self.conv4_2(x)
    detection_sources.append(x) # (32, 256, 8, 8)

    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())

    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))

    mimic = self.avgpool(x).view(x.size(0), -1)
    return x, mimic, output
