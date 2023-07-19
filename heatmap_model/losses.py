import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn

import math
    
class FocalLoss_interaction(nn.Module):
    def __init__(self, para, weight=None, size_average=True):
        self.para = para
        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.sigmax = para['sigmax']
        self.sigmay = para['sigmay']
        self.dx, self.dy = para['resolution'], para['resolution']
        
        lateral = torch.tensor([i for i in range(int(-self.xmax/self.dx), 
                                                     int(self.xmax/self.dx)+1)])*self.dx
        longitudinal = torch.tensor([i for i in range(int(self.ymin/self.dy), 
                                                     int(self.ymax/self.dy)+1)])*self.dy

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        self.x = lateral.repeat(self.len_y, 1).transpose(1,0)
        self.y = longitudinal.repeat(self.len_x, 1)
        super(FocalLoss_interaction, self).__init__()
        
    def bce(self, yp, y):
        loss = y*torch.log(yp)+(1.-y)*torch.log(1.-yp)
        return -torch.sum(loss)
    
    def forward(self, inputs, targets, alpha=0.25, gamma=2., smooth=1):
        inputs = inputs.float()
        ref = torch.zeros_like(inputs)
        for i in range(ref.size(0)):
            xc = torch.ones_like(self.x)*targets[i,0].item()
            yc = torch.ones_like(self.y)*targets[i,1].item()
            ref[i] = torch.exp(-((self.x-xc)**2/self.sigmax**2)/2 - ((self.y-yc)**2/self.sigmay**2)/2)
        
        inputs = inputs.reshape(-1)
        ref = ref.reshape(-1)
        BCE = F.binary_cross_entropy_with_logits(inputs, ref.float(), reduction='sum')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
class OverAllLoss(nn.Module):
    def __init__(self, para):
        super(OverAllLoss, self).__init__()
        self.heatmap_loss = FocalLoss_interaction(para)
    def forward(self, inputs, targets):
        lmain = self.heatmap_loss(inputs[0], targets[0])
        bce = F.binary_cross_entropy_with_logits(inputs[1].reshape(-1), targets[1].reshape(-1), reduction='sum')
        bce_exp = torch.exp(-bce)
        lreg = 0.25 * (1-bce_exp)**2 * bce
        
        return 3*lmain + lreg
        #return lreg

class OverAllLoss_reg(nn.Module):
    def __init__(self, para):
        super(OverAllLoss_reg, self).__init__()
        self.heatmap_loss = FocalLoss_interaction(para)
        self.l1_loss = nn.L1Loss(reduction='mean')
    def forward(self, inputs, targets, lmbd=1e4):
        lmain = self.heatmap_loss(inputs[0], targets[0])
        hreg = self.heatmap_loss(inputs[2], targets[0])
        bce = F.binary_cross_entropy_with_logits(inputs[1].reshape(-1), targets[1].reshape(-1), reduction='sum')
        bce_exp = torch.exp(-bce)
        lreg = 0.25 * (1-bce_exp)**2 * bce
        
        coefficient = self.l1_loss(torch.sigmoid(inputs[0]), torch.sigmoid(inputs[2]))*lmbd+1
        #print(coefficient.item(), lmain.item(), lreg.item()/3, hreg.item())
        return (lmain + lreg/15)*coefficient + hreg