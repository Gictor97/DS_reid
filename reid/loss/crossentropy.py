import torch
import torch.nn as nn
from pdb import set_trace

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, t_class=0,epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.t_class = t_class
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        N = targets.size(0)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / (self.num_classes+self.t_class)
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class domain_regluarization_loss(nn.Module):
    '''
    cross domain loss regularizer
    regularizer domain confusion when  jointly training
    将对应的数据实例分类正确的mask掉，剩余的作为错误判别项
    input:
    s_class,t_class:the num of source domain and target domain
    labels:the labels
    '''

    def __init__(self, num_classes=2048, batch_size=64):
        super(domain_regluarization_loss, self).__init__()
        self.num_classes = num_classes
        self.batch = batch_size
    def forward(self, s_class,t_class,labels, probs,N=1,idm_lam=None,mode='all'):
        self.s_class = s_class
        self.t_class = t_class
        self.softmax = nn.Softmax(dim=1).cuda()
        if isinstance(idm_lam,torch.Tensor):
            label_s,label_t = labels.split(labels.size(0)//2,dim=0)
            label_mix = idm_lam[:,0]*label_s + idm_lam[:,1]+label_t
            labels = torch.cat((label_s,label_mix,label_t),dim=0)
        assert labels.size()[0] == probs.size()[0]
        probs = self.softmax(probs)
        _,index = torch.topk(probs,N,dim=1)
        if mode == 'mask':
            mask = torch.ones_like(probs).scatter_(1,index,1)
        elif mode == 'all':
            mask = torch.ones_like(probs)
        else:
            raise LookupError('mode of regularition error')
        if  isinstance(idm_lam,torch.Tensor):
            mask[0:self.batch, 0:self.s_class] = 0  #s_Data
            mask[self.batch:self.batch * 2, 0:self.s_class + self.t_class] = 0 #mix
            mask[self.batch*2:,self.s_class:self.s_class+self.t_class]=0  #tar_data
        else:
            mask[0:self.batch, 0:self.s_class] = 0
            mask[self.batch:self.batch * 2, self.s_class:self.s_class + self.t_class] = 0
        probs = probs*mask


        loss = probs.sum(dim =1).mean()

        return loss
