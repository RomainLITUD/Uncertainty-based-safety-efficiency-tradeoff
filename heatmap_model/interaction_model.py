from heatmap_model.baselayers import *
from heatmap_model.encoder import *
from heatmap_model.decoder import *

class CTnet(nn.Module):
    def __init__(self, para):
        super(CTnet, self).__init__()
        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.resolution = para['resolution']
        self.test = para['test']
        
        self.encoder = VectorEncoder(para)
        self.decoder = VectorDecoder(para)
        
        lateral = torch.tensor([i for i in range(int(-self.xmax/self.resolution), 
                                                         int(self.xmax/self.resolution)+1)])*self.resolution
        longitudinal = torch.tensor([i for i in range(int(self.ymin/self.resolution), 
                                                     int(self.ymax/self.resolution)+1)])*self.resolution

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        x1 = lateral.repeat(self.len_y, 1).transpose(1,0)
        y1 = longitudinal.repeat(self.len_x, 1)
        self.mesh = nn.Parameter(torch.stack((x1,y1),-1),requires_grad = False)
        self.tmesh = nn.Parameter(torch.tensor([i+1 for i in range(30)]).unsqueeze(-1), requires_grad = False)
       
    def forward(self, trajectory, maps, lanefeatures, adj, af, c_mask, timestamp, gtxy):
        hlane, hmid, hinteraction = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        grid = self.mesh.reshape(-1, 2)

        #gtcoor = gtxy.unsqueeze(-2).repeat(1, 1, 30, 1)
        
        gttime = self.tmesh.unsqueeze(0).unsqueeze(0).repeat(gtxy.size(0), gtxy.size(1), 1, 1)
        #print(gttime.size(), gtxy.size())
        gtpoint = torch.cat((gtxy, gttime),-1).reshape(gtxy.size(0), -1, 3)

        heatmap = self.decoder(hlane, hmid, hinteraction, grid, c_mask, timestamp, gtpoint)
        if not self.test:
            sl = 30*gtxy.size(-3)
            #print(sl, heatmap.size())
            tslice = heatmap[:,-sl:].reshape(maps.size(0), gtxy.size(-3), 30)
            heatmap = heatmap[:,:-sl].reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))

            return heatmap, tslice
        else:
            heatmap = heatmap.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
            return torch.sigmoid(heatmap).squeeze()
        
    def forward_old(self, trajectory, maps, lanefeatures, adj, af, c_mask, timestamp, gtxy):
        hlane, hmid, hinteraction = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)

        heatmap = self.decoder(hlane, hmid, hinteraction, gtxy, c_mask, timestamp)
        return torch.sigmoid(heatmap).squeeze()
        
        
class CTnet_causal(nn.Module):
    def __init__(self, para):
        super(CTnet_causal, self).__init__()
        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.resolution = para['resolution']
        self.test = para['test']
        
        self.encoder = VectorEncoder(para)
        self.decoder = VectorDecoder(para)
        self.reg_decoder = RegularizeDecoder(para)
        
        lateral = torch.tensor([i for i in range(int(-self.xmax/self.resolution), 
                                                         int(self.xmax/self.resolution)+1)])*self.resolution
        longitudinal = torch.tensor([i for i in range(int(self.ymin/self.resolution), 
                                                     int(self.ymax/self.resolution)+1)])*self.resolution

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        x1 = lateral.repeat(self.len_y, 1).transpose(1,0)
        y1 = longitudinal.repeat(self.len_x, 1)
        self.mesh = nn.Parameter(torch.stack((x1,y1),-1),requires_grad = False)
        self.tmesh = nn.Parameter(torch.tensor([i+1 for i in range(30)]).unsqueeze(-1), requires_grad = False)
       
    def forward(self, trajectory, maps, lanefeatures, adj, af, c_mask, timestamp, gtxy):
        hlane, hmid, hinteraction = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        grid = self.mesh.reshape(-1, 2)

        #gtcoor = gtxy.unsqueeze(-2).repeat(1, 1, 30, 1)
        gttime = self.tmesh.unsqueeze(0).unsqueeze(0).repeat(gtxy.size(0), gtxy.size(1), 1, 1)
        gtpoint = torch.cat((gtxy, gttime),-1).reshape(gtxy.size(0), -1, 3)

        heatmap = self.decoder(hlane, hmid, hinteraction, grid, c_mask, timestamp, gtpoint)
        h_reg = self.reg_decoder(torch.cat((hlane, hmid[:,55:56]), 1), grid, c_mask, adj, timestamp)
        
        #sl = 30*gtxy.size(-3)
        #tslice = heatmap[:,-sl:].reshape(maps.size(0), gtxy.size(-3), 30)
        #heatmap = heatmap[:,:-sl].reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
        heatmap = heatmap.reshape(self.mesh.size(0), self.mesh.size(1))
        
        
        #return heatmap, tslice, h_reg.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
        return torch.sigmoid(heatmap), torch.sigmoid(h_reg).reshape(self.mesh.size(0), self.mesh.size(1))

