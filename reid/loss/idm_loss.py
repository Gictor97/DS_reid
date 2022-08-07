from __future__ import absolute_import

import torch
from torch import nn


class DivLoss(nn.Module):
    def __init__(self, ):
        super(DivLoss, self).__init__()
        
    def forward(self, scores):
        mu = scores.mean(0)
        std = ((scores-mu)**2).mean(0,keepdim=True).clamp(min=1e-12).sqrt()
        loss_std = -std.sum()
        return loss_std

class BridgeFeatLoss(nn.Module):
    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, feats_s, feats_t, feats_mixed, lam):

        dist_mixed2s = ((feats_mixed-feats_s)**2).sum(1, keepdim=True)
        dist_mixed2t = ((feats_mixed-feats_t)**2).sum(1, keepdim=True)

        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()

        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (lam*dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()

        return loss


class BridgeProbLoss(nn.Module):
    '''
    input :prediction tensor (batch_size,num_classes)   (192,12938)
    target:lanbel tensor(batch_size)      (128,11)
    lam:weight(batch_size,1)            (64,1)  only weight from source dataset
    '''
    def __init__(self, num_classes, epsilon=0.1):
        super(BridgeProbLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.device_num = torch.cuda.device_count()

    def forward(self, inputs, targets, lam):


        inputs = inputs.view(self.device_num, -1, inputs.size(-1))          #(2,96,12938)
        inputs_s,inputs_mixed ,inputs_t = inputs.split(inputs.size(1) // 3, dim=1)   #(2,32,12938)
        inputs_ori = torch.cat((inputs_s,inputs_t),1).view(-1,inputs.size(-1))   ##(128,12938)
        inputs_mixed = inputs_mixed.contiguous().view(-1, inputs.size(-1))          #(64,12938)
        log_probs_ori = self.logsoftmax(inputs_ori)
        log_probs_mixed = self.logsoftmax(inputs_mixed)

        targets = torch.zeros_like(log_probs_ori).scatter_(1, targets.unsqueeze(1), 1)
        targets = targets.view(self.device_num, -1, targets.size(-1))
        targets_s, targets_t = targets.split(targets.size(1) // 2, dim=1)


        targets_s = targets_s.contiguous().view(-1, targets.size(-1))
        targets_t = targets_t.contiguous().view(-1, targets.size(-1))

        targets = targets.view(-1, targets.size(-1))
        soft_targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes  #soft label

        lam = lam.view(-1, 1)
        soft_targets_mixed = lam*targets_s+(1.-lam)*targets_t
        soft_targets_mixed = (1 - self.epsilon) * soft_targets_mixed + self.epsilon / self.num_classes
        loss_ori = (- soft_targets*log_probs_ori).mean(0).sum()
        loss_bridge_prob = (- soft_targets_mixed*log_probs_mixed).mean(0).sum()   ##mixed domain use mixed label with a weight of lam

        return loss_ori, loss_bridge_prob# 返回初始交叉熵以及混合交叉熵


