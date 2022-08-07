# -*- coding: utf-8 -*-
# @Time   : 2021/6/14 - 12:26
# @File   : gem_torch.py
# @Author : surui

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeMPooling(nn.Module):
    def __init__(self, feature_size, pool_size=7, init_norm=3.0, eps=1e-6, normalize=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.feature_size = feature_size  # Final layer channel size, the pow calc at -1 axis
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.p = torch.nn.Parameter(torch.ones(self.feature_size) * self.init_norm, requires_grad=True)
        self.p.data.fill_(init_norm)
        self.normalize = normalize
        self.avg_pooling = nn.AvgPool2d((self.pool_size, self.pool_size))
        self.eps = eps

    def forward(self, features):
        # filter invalid value: set minimum to 1e-6
        # features-> (B, H, W, C)
        features = features.clamp(min=self.eps).pow(self.p)
        features = features.permute((0, 3, 1, 2))
        features = self.avg_pooling(features)
        features = torch.squeeze(features)
        features = features.permute((0, 2, 3, 1))
        features = torch.pow(features, (1.0 / self.p))
        # unit vector
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        return features


if __name__ == '__main__':
    x = torch.randn((8, 7, 7, 768)) * 0.02
    p = torch.nn.Parameter(torch.FloatTensor(1))
    gem = GeneralizedMeanPooling(p)

    print("input : ", x)
    print("=========================")
    print(gem(x).size())
