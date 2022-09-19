from __future__ import absolute_import

from .triplet import TripletLoss
from .triplet_xbm import TripletLossXBM
from .crossentropy import CrossEntropyLabelSmooth,domain_regluarization_loss
from .idm_loss import DivLoss, BridgeFeatLoss, BridgeProbLoss
from .mmd_loss import  DDC_loss,mix_mmdaloss
__all__ = [
    'DivLoss',
    'BridgeFeatLoss',
    'BridgeProbLoss',
    'TripletLoss',
    'TripletLossXBM',
    'CrossEntropyLabelSmooth',
    'DDC_loss'
]
