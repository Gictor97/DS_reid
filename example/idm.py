from reid import models
import torch
import os
from pdb import  set_trace
import pynvml


if __name__=='__main__':

    devices = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet50_idm(num_classes = 13682).to(devices)
    data = torch.Tensor(128,3,256,128).to(devices)
    a,b ,lam= model(data)
    print(model.base[1])
    print(a.size(),b.size())

