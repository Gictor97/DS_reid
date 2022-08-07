from __future__ import print_function
from torch import nn
import torch
from .modules import gempooling
from pdb import set_trace
class  GCPooling(nn.Module):
    def __init__(self,channel =64):
        super(GCPooling,self).__init__()
        self.conv1=nn.Conv2d(channel,channel,1)
        self.conv2 =nn.Sequential(
            nn.Conv2d(channel*2,channel,1),
            nn.BatchNorm2d(num_features=channel),
            nn.ReLU(inplace=True)
        )
        self.channel =channel
    def forward(self,max,avg):
        if (not self.training):
            return max,avg
        p_cont =self.conv1(avg-max)   ##(bs,channel,1)
        p_max = self.conv1(max)
        p_con_max = torch.cat((p_cont,p_max),dim=1)
        p_conmax = self.conv2(p_con_max)
        Gcp = p_conmax+max
        return Gcp.squeeze()

class GPP_pooling(nn.Module):
    def __init__(self, part= 3,channel=64):
        super(GPP_pooling, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(num_features=channel),
            nn.ReLU(inplace=True)
        )
        self.channel = channel
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.avepool2 = nn.AdaptiveAvgPool2d((part, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        avg = self.avepool(x).squeeze(-1)
        avg2 = self.avepool2(x)
        max1 = self.maxpool(x)
        p_cot = self.conv1(avg2 - max1.expand_as(avg2)).squeeze()
        p_cot_avg = torch.cat((avg, p_cot), dim=-1)
        gpp = self.conv2(p_cot_avg.unsqueeze(-1))
        return gpp.squeeze().mean(-1)

class idm(nn.Module):
    def __init__(self,channel = 64):### channel means the num of feature channel every picture
        super(idm,self).__init__()
        self.avepool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.gpp = GPP_pooling(channel=channel)
        self.softmax = nn.Softmax(dim=1)  ##keep lam_s+lam_t =1
        self.FC1 = nn.Linear(channel * 2, channel)
        self.FC2 = nn.Linear(channel, int(channel / 2))
        self.FC3 = nn.Linear(int(channel / 2), 2)
      #  self.register_parameter('p',torch.Tensor([3,5]))
        self.p = nn.Parameter(torch.tensor([3.,5.]))
        self.gempooling_s = gempooling(self.p[0])
        self.gempooling_t = gempooling(self.p[1])
        self.modelist =['normal','max_gem','ave_gem','gem','gcp','gpp']
        self.Gcp_ds= GCPooling(channel)
        self.Gcp_dt= GCPooling(channel)

    def forward(self,x,mode='normal'):
        if (not self.training):
            return x
        b ,channel,_,_= x.size()
        assert (b%2==0)
        split = torch.split(x,int(b/2),0)
        ds = split[0].contiguous().detach()
        dt = split[1].contiguous().detach()
        if mode =='normal':
            ds_pool = torch.cat((self.avepool(ds).squeeze(),self.maxpool(ds).squeeze()),1)              #(batch_size,2channel,1)
            dt_pool = torch.cat((self.avepool(dt).squeeze(),self.maxpool(dt).squeeze()),1)
        elif mode ==self.modelist[1]:
            ds_pool = torch.cat((self.gempooling_s(ds).squeeze(), self.maxpool(ds).squeeze()), 1)  # (batch_size,2channel,1)
            dt_pool = torch.cat((self.gempooling_t(dt).squeeze(), self.maxpool(dt).squeeze()), 1)
        elif mode == self.modelist[2]:
            ds_pool = torch.cat((self.avepool(ds).squeeze(), self.gempooling_s(ds).squeeze()), 1)  # (batch_size,2channel,1)
            dt_pool = torch.cat((self.avepool(dt).squeeze(), self.gempooling_t(dt).squeeze()), 1)
        elif mode == self.modelist[3]:
            ds_pool = torch.cat((self.gempooling_s(ds).squeeze(), self.gempooling_s(ds).squeeze()),1)  # (batch_size,2channel,1)
            dt_pool = torch.cat((self.gempooling_t(dt).squeeze(), self.gempooling_t(dt).squeeze()), 1)
        elif mode == self.modelist[4]:
            ds_pool = self.Gcp_ds(self.maxpool(ds).squeeze(1),self.avepool(ds).squeeze(1))
            dt_pool = self.Gcp_dt(self.maxpool(dt).squeeze(1),self.avepool(dt).squeeze(1))
        elif mode == self.modelist[5]:
            ds_pool =self.gpp(ds)
            dt_pool =self.gpp(dt)  #(64,channel)

        else:
            raise Exception('mode can not match')

        ##(64,channel*2)
        if ds_pool.size(1)==channel*2:
            ds_fc = self.FC1(ds_pool)
            dt_fc = self.FC1(dt_pool)
        else:
            ds_fc = ds_pool##(64,64)
            dt_fc = dt_pool
        x_embed = ds_fc+dt_fc   #(bs,c)
        x_embed = self.FC2(x_embed)
        embed = self.FC3(x_embed)
        embed = self.softmax(embed)  ###(64,2)

        inter = embed[:,0].reshape(-1,1,1,1)*ds+embed[:,1].reshape(-1,1,1,1)*dt
        out = torch.cat((ds,inter,dt),0)

        return embed,out



if __name__ == "__main__":
    data_A = torch.Tensor(64,64,64,32)
    data_B =torch.Tensor(64,64,64,32)
    idms= idm()
    gather = torch.cat((data_A,data_B))
    embed ,out = idms(gather)
    print(embed.size(),out.size())