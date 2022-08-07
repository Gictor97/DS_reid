import torch
import torch.nn as nn
from einops.layers.torch import Rearrange,Reduce


class Smooth_scale_Module(nn.Module):
    def __init__(self,channel_list =[256,512,1024],fix_channel = 2048 ):
        super(Smooth_scale_Module,self).__init__()
        self.con1_c1 = nn.Conv2d(channel_list[0],fix_channel,1)
        self.con1_c2 = nn.Conv2d(channel_list[1],fix_channel,1)
        self.con1_c3 = nn.Conv2d(channel_list[3],fix_channel,1)
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self,x1,x2,x3,blocks):
        f1 = self.gap(self.con1_c1(x1))
        f2 = self.gap(self.con1_c2(x2))
        f3 = self.gap(self.con1_c3(x3))
        f1 = Rearrange(f1,'b c h w ->block b c h/block w',block = blocks)
        f2 = Rearrange(f2,'b c h w ->block b c h/block w',block = blocks)
        f3 = Rearrange(f3,'b c h w ->block b c h/block w',block = blocks)



