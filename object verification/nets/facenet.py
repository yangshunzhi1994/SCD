import torch.nn as nn
from torch.nn import functional as F
from nets.studentNet import CNN_RIS
from nets.teacherNet import Teacher
        
class Facenet(nn.Module):
    def __init__(self, backbone, dropout_keep_prob=0.5, embedding_size=2048, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == "Teacher":
            self.backbone = Teacher(ResNet_factor=4, num_classes=num_classes)
        elif backbone == "Teacher_DGKD1":
            self.backbone = Teacher(ResNet_factor=2, num_classes=num_classes)
        elif backbone == "Teacher_DGKD2":
            self.backbone = Teacher(ResNet_factor=0, num_classes=num_classes)
        else:
            self.backbone = CNN_RIS(num_classes=num_classes)
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(num_classes, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode="predict"):
        if mode == 'predict':
            rb1, rb2, rb3, feat, mimic, x = self.backbone(x)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        rb1, rb2, rb3, feat, mimic, x = self.backbone(x)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return feat, mimic, x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x
