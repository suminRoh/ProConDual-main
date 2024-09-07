import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

        
class ProDual(nn.Module):

    def __init__(self, tau=1, weight=None, batch_size=256, num_classes=2):
        super(ProDual, self).__init__()
    def forward(self, centers, classifier_weight):

        cw_norm=torch.norm(classifier_weight.T, p='fro')
        cp_norm=torch.norm(centers, p='fro')
        
        cw=classifier_weight.T/cw_norm
        cp=centers/cp_norm
        
        sub=cw-cp
        loss=torch.norm(sub, p='fro')

        return loss        
         