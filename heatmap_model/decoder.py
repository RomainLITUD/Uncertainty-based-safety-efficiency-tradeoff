import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from heatmap_model.utils import *
from heatmap_model.baselayers import *

import math

class VectorDecoder(nn.Module):
    def __init__(self, para):
        super(VectorDecoder, self).__init__()
        self.para = para
        self.hidden_size = para['encoder_attention_size']
        
        self.ego2coor = EgoAssign(self.hidden_size)
        self.l2c = MultiHeadCrossAttention(self.hidden_size,self.hidden_size,self.hidden_size//2, 2)
        self.l2c2 = MultiHeadCrossAttention(self.hidden_size,self.hidden_size,self.hidden_size//2, 2)
        c_h = self.hidden_size
        self.convert = DecoderResCat(4*c_h, c_h, 1)

    def forward(self, hlane, hmid, hinteraction, coordinates, c_mask, timestamp, gtpoint):
        s_order = torch.stack([torch.ones_like(coordinates[...,:1])*s for s in timestamp])            
        coords = coordinates.unsqueeze(0).repeat(hlane.size(0), 1, 1)
        coords = torch.cat((coords, s_order), -1) #(h,N,3)

        if not self.para['test']:
            coords = torch.cat((coords, gtpoint), -2)
        
        position1 = self.ego2coor(coords, hinteraction[:,55:56])
        position2 = self.l2c(position1, hmid, c_mask)
        position3 = self.l2c2(position1, hlane, c_mask[:,:55])
        #position3 = self.l2c2(position1, hinteraction, c_mask)
        li = torch.cat((hinteraction[:,55:56].repeat(1,position1.size(1), 1), position1, position2, position3),-1)
        
        heatmap = self.convert(li).squeeze()
        return heatmap
    

    def forward_old(self, hlane, hmid, hinteraction, coordinates, c_mask, timestamp):
        coords = torch.cat((coordinates, torch.ones_like(coordinates[...,:1])*timestamp), -1) #(h,N,3)
 
        position1 = self.ego2coor(coords, hinteraction[:,55:56])
        position2 = self.l2c(position1, hmid, c_mask)
        position3 = self.l2c2(position1, hlane, c_mask[:,:55])
        li = torch.cat((hinteraction[:,55:56].repeat(1,position1.size(1), 1), position1, position2, position3),-1)
        
        heatmap = self.convert(li).squeeze()
        return heatmap
      
class RegularizeDecoder(nn.Module):
    def __init__(self, para):
        super(RegularizeDecoder, self).__init__()
        self.hidden_size = para['encoder_attention_size']
        
        self.map2ego = MultiHeadSelfAttention(self.hidden_size, self.hidden_size//2, 2)
        self.lnorm = LayerNorm(self.hidden_size)
        
        c_h = self.hidden_size
        self.convert = DecoderResCat(c_h, c_h, 1)
        self.act = nn.ReLU()
        self.heatmapdecoder = ToCoordinateCrossAttention(self.hidden_size, self.hidden_size//2, 2)

    def forward(self, hmae, coordinates, c_mask, adj, timestamp):
        s_order = torch.stack([torch.ones_like(coordinates[...,:1])*s for s in timestamp])
        coords = coordinates.unsqueeze(0).repeat(hmae.size(0), 1, 1)
        #print(s_order.size(), coords.size())
        coords = torch.cat((coords, s_order), -1) #(h,N,3)
        
        #print(hego.size(), hmap.size())
        #h = self.map2ego(hego, hmap, c_mask[:55])
        h = self.map2ego(hmae, adj[:,:56,:56])
        h = self.act(h)
        h = h+hmae
        h = self.lnorm(h)
        
        h = torch.cat((h[:,:55], hmae[:,55:56]),1)
        
        h = self.heatmapdecoder(h, coords, c_mask[:,:56])
        heatmap = self.convert(h).squeeze()

        return heatmap