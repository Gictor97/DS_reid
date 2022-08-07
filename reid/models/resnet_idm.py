from __future__ import absolute_import
from pdb import set_trace
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import sys

from .idm_module import idm
from ..utils.logging import Logger
from collections import OrderedDict

__all__ = ['ResNet_idm', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet_idm(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }
    __channel = [64, 256, 512, 1024, 2056]

    def __init__(self, depth, pool='normal',pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, stage=0):
        super(ResNet_idm, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.pool = pool
        # Construct base (pretrained) resnet
        if depth not in ResNet_idm.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_idm.__factory[depth](pretrained=pretrained)  ###加载模型,会将参数模型一起下载

        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.stage = stage
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', resnet.conv1),
            ('bn1', resnet.bn1),
            ('relu', resnet.relu),
            ('maxpool', resnet.maxpool)]))

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.base = nn.ModuleList(
            [self.conv,
             self.layer1,
             self.layer2,
             self.layer3,
             self.layer4],

        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        if 0 <= self.stage < len(self.__channel):
            self.idm = idm(self.__channel[self.stage])
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0  ##false-
            self.num_classes = num_classes
            out_planes = resnet.fc.in_features
            # Append new layers,
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        for m in self.idm.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
        if not pretrained:
            print("reset_parameter")
            self.reset_params()

    def forward(self, x, output_prob=False):
        bs = x.size(0)

        for i in range(5):
            net = self.base[i]
            x = net(x)

            if self.stage == i and self.training:
                lam, x = self.idm(x,self.pool)


        x = self.gap(x)
        x = x.view(x.size(0), -1)  ###(bs*2,outplanes)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False and output_prob is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            norm_bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)

        else:
            return bn_x

        if self.norm:
            return prob, x, norm_bn_x, lam
        else:
            return prob, x, lam    ##predictions,featureas map,weight

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def resnet18_idm(**kwargs):
    return ResNet_idm(18, **kwargs)


def resnet34_idm(**kwargs):
    return ResNet_idm(34, **kwargs)


def resnet50_idm(**kwargs):
    return ResNet_idm(50, **kwargs)


def resnet101_idm(**kwargs):
    return ResNet_idm(101, **kwargs)


def resnet152_idm(**kwargs):
    return ResNet_idm(152, **kwargs)
