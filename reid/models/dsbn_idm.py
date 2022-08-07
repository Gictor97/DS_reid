import torch
import torch.nn as nn

# Domain-specific BatchNorm
##把输入的数据分成源域和目标域分别输入到不同的BN层。
##旨在训练层做dsbn,测试时
class DSBN2d(nn.Module):
    def __init__(self, planes):
        super(DSBN2d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())

        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out

class DSBN1d(nn.Module):
    def __init__(self, planes):
        super(DSBN1d, self).__init__()
        self.num_features = planes
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)

    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        assert (bs%2==0)
        split = torch.split(x, int(bs/2), 0)
        out1 = self.BN_S(split[0].contiguous())
        out2 = self.BN_T(split[1].contiguous())
        out = torch.cat((out1, out2), 0)
        return out



class DSBN1d_idm(nn.Module):
    def __init__(self,planes):
        super(DSBN1d_idm, self).__init__()
        self.planes =planes
        self.mix = nn.BatchNorm1d(planes)
        self.BN_S = nn.BatchNorm1d(planes)
        self.BN_T = nn.BatchNorm1d(planes)
    def forward(self, x):
        if (not self.training):
            return self.BN_T(x)

        bs = x.size(0)
        if  (bs % 3 == 0):
            split = torch.split(x, int(bs / 3), 0)
            out1 = self.BN_S(split[0].contiguous())
            out3 = self.mix(split[1].contiguous())
            out2 = self.BN_T(split[2].contiguous())
            out = torch.cat((out1,out3,out2), 0)
        elif(bs%2==0):
            split = torch.split(x, int(bs / 2), 0)
            out1 = self.BN_S(split[0].contiguous())
            out2 = self.BN_T(split[1].contiguous())
            out = torch.cat((out1, out2), 0)
        return out


class DSBN2d_idm(nn.Module):
    def __init__(self,planes):
        super(DSBN2d_idm, self).__init__()
        self.planes = planes
        self.mix = nn.BatchNorm2d(planes)
        self.BN_S = nn.BatchNorm2d(planes)
        self.BN_T = nn.BatchNorm2d(planes)
    def forward(self, x):

        if (not self.training):
            return self.BN_T(x)
        bs = x.size(0)
        if  (bs % 3 == 0):
            split = torch.split(x, int(bs / 3), 0)
            out1 = self.BN_S(split[0].contiguous())
            out3 = self.mix(split[1].contiguous())
            out2 = self.BN_T(split[2].contiguous())
            out = torch.cat((out1,out3,out2), 0)
        elif(bs%2==0):
            split = torch.split(x, int(bs / 2), 0)
            out1 = self.BN_S(split[0].contiguous())
            out2 = self.BN_T(split[1].contiguous())
            out = torch.cat((out1, out2), 0)
        return out

def convert_dsbn(model):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, nn.BatchNorm2d):
            m = DSBN2d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())
            setattr(model,child_name,m)
        elif isinstance(child, nn.BatchNorm1d) and child_name!='d_bn1':
            m = DSBN1d(child.num_features)
            m.BN_S.load_state_dict(child.state_dict())
            m.BN_T.load_state_dict(child.state_dict())

            setattr(model, child_name, m)
        else:
            convert_dsbn(child)

def convert_dsbn_idm(model):
    for _,(child_name,child) in enumerate(model.named_children()):
        #assert(not(next(model.parameters().is_cuda)))
        if child_name not in ['BN_S','BN_T','mix']:
            if isinstance(child,nn.BatchNorm2d):
                m = DSBN2d_idm(child.num_features)
                m.BN_S.load_state_dict(child.state_dict())
                m.BN_T.load_state_dict(child.state_dict())
                m.mix.load_state_dict(child.state_dict())
                setattr(model,child_name,m)
            elif isinstance(child,nn.BatchNorm1d):
                m = DSBN1d_idm(child.num_features)
                m.BN_S.load_state_dict(child.state_dict())
                m.BN_T.load_state_dict(child.state_dict())
                m.mix.load_state_dict(child.state_dict())
                setattr(model,child_name,m)
            else:
                convert_dsbn_idm(child)


def convert_bn(model, use_target=True):
    for _, (child_name, child) in enumerate(model.named_children()):
        assert(not next(model.parameters()).is_cuda)
        if isinstance(child, DSBN2d):
            m = nn.BatchNorm2d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        elif isinstance(child, DSBN1d):
            m = nn.BatchNorm1d(child.num_features)
            if use_target:
                m.load_state_dict(child.BN_T.state_dict())
            else:
                m.load_state_dict(child.BN_S.state_dict())
            setattr(model, child_name, m)
        else:
            convert_bn(child, use_target=use_target)
