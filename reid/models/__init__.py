from __future__ import absolute_import
from .resnet import *
from .resnet_ibn import *
from .dsbn import convert_bn,convert_dsbn
from .dsbn_idm import convert_dsbn_idm
from .idm_module import *
from .resnet_idm import resnet50_idm
__factory = {
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet_50a':resnet_ibn50a,
    'resnet50_idm':resnet50_idm

}
def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a models instance.

    Parameters
    ----------
    name : str
        Model name. Can be one of 'inception', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', and 'resnet152'.
    pretrained : bool, optional
        Only applied for 'resnet*' models. If True, will use ImageNet pretrained
        models. Default: True
    cut_at_pooling : bool, optional
        If True, will cut the models before the last global pooling layer and
        ignore the remaining kwargs. Default: False
    num_features : int, optional
        If positive, will append a Linear layer after the global pooling layer,
        with this number of output units, followed by a BatchNorm layer.
        Otherwise these layers will not be appended. Default: 256 for
        'inception', 0 for 'resnet*'
    norm : bool, optional
        If True, will normalize the feature to be unit L2-norm for each sample.
        Otherwise will append a ReLU layer after the above Linear layer if
        num_features > 0. Default: False
    dropout : float, optional
        If positive, will append a Dropout layer with this dropout rate.
        Default: 0
    num_classes : int, optional
        If positive, will append a Linear layer at the end as the classifier
        with this number of output units. Default: 0
    """
    if name not in __factory:
        raise KeyError("Unknown models:", name)
    return __factory[name](*args, **kwargs)
