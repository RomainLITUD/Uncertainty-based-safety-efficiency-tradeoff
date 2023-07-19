import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
from scipy.sparse import csr_matrix
from skimage.transform import rescale
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InteractionDataset(Dataset):
    """
    filename: a list of files or one filename of the .npz file
    stage: {"train", "val", "test"}
    """
    def __init__(self, filenames, stage, para):
        self.stage= stage
        self.para = para
        self.step = para['step']
        self.resolution = para['resolution']
        self.sigmax = para['sigmax']
        self.sigmay = para['sigmay']
        if stage == 'train':
            self.T = []
            self.M = []
            self.L = []
            self.N_agents = []
            self.N_splines = []
            self.Adj = []
            for filename in filenames:
                data = np.load('./interaction_merge/'+filename+'.npz', allow_pickle=True)

                self.T.append(data['trajectory'])
                self.M.append(data['maps'])
                self.L.append(data['lanefeature'])
                self.N_agents.append(data['nbagents'])
                self.N_splines.append(data['nbsplines'])
                self.Adj.append(data['adjacency'])

            self.T = np.concatenate(self.T, axis=0)
            self.M = np.concatenate(self.M, axis=0)
            self.L = np.concatenate(self.L, axis=0)
            self.N_agents = np.concatenate(self.N_agents, axis=0)
            self.N_splines = np.concatenate(self.N_splines, axis=0)
            self.Adj = np.concatenate(self.Adj, 0)
                
            data = np.load('./interaction_merge/traj_train.npz', allow_pickle=True)
            self.Y = data['traj']
        else:
            data = np.load('./interaction_merge/'+filenames[0]+'.npz', allow_pickle=True)
            self.T = data['trajectory']
            self.M = data['maps']
            self.L = data['lanefeature']
            self.N_agents = data['nbagents']
            self.N_splines = data['nbsplines']
            self.Adj = data['adjacency']

            if stage=='val':
                data = np.load('./interaction_merge/traj_val.npz', allow_pickle=True)
                self.Y = data['traj']
        
    def __len__(self):
        return len(self.N_agents)
    
    def __getitem__(self, index):
        traj = torch.tensor(self.T[index]).float().to(device)
        splines = torch.tensor(self.M[index]).float().to(device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(device)
        nb_agents = self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        a = self.Adj[index].toarray()
        af = a.copy()#+np.eye(55)
        af[af<0] = 0
        pad = np.zeros((55,55))
        pad[:nb_splines,:nb_splines] = np.eye(nb_splines)

        Af = np.linalg.matrix_power(af+pad+af.T, 4)
        Af[Af>0]=1

        A_f = torch.Tensor(Af).float().to(device)

        adjacency = np.zeros((81, 81))
        adjacency[:nb_splines][...,:nb_splines] = 1
        adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
        adjacency[:nb_splines][...,55:55+nb_agents] = 1
        adjacency[55:55+nb_agents][...,:nb_splines] = 1
        adj = torch.Tensor(adjacency).int().to(device)

        c_mask = torch.Tensor(adjacency[:,0]).int().to(device)
            
        v = 30#random.randint(1, 30)
        if self.stage != "test":
            y_ = self.Y[index]
            y = torch.Tensor(y_[v-1]).float().to(device)
            timestamp = torch.tensor(v).int().to(device)

            stepall = 2*self.step+1
            
            xc, yc = y_[:,0], y_[:,1]
            mx = np.linspace(xc-self.step*self.resolution, xc+self.step*self.resolution, stepall)
            my = np.linspace(yc-self.step*self.resolution, yc+self.step*self.resolution, stepall)
            
            corx = np.tile(mx, (stepall,1,1)).transpose(1,0,2)
            cory = np.tile(my, (stepall,1,1))
            gtcoor = np.stack([corx, cory],-1).reshape(-1,30,2)
            gttime = np.array([np.exp(-((gx[:,0]-xc)**2/self.sigmax**2)/2 - ((gx[:,1]-yc)**2/self.sigmay**2)/2) for gx in gtcoor])

            gtcoor = torch.tensor(gtcoor).float().to(device)
            gttime = torch.tensor(gttime).float().to(device)
            
            return traj, splines, lanefeature, adj, A_f, c_mask, timestamp, y, gtcoor, gttime
        else:
            timestamp = torch.tensor(v).int().to(device)
            gtxy = torch.zeros((2*self.step+1)**2, 2).int().to(device)
            return traj, splines, lanefeature, adj, A_f, c_mask, timestamp, gtxy
        
    def test_sampling(self, index, moment):
        traj = torch.tensor(self.T[index]).float().to(device)
        splines = torch.tensor(self.M[index]).float().to(device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(device)
        nb_agents = self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        a = self.Adj[index].toarray()
        af = a.copy()#+np.eye(55)
        af[af<0] = 0
        pad = np.zeros((55,55))
        pad[:nb_splines,:nb_splines]=np.eye(nb_splines)

        Af = np.linalg.matrix_power(af+pad+af.T, 4)
        Af[Af>0]=1

        A_f = torch.Tensor(Af).float().to(device)

        adjacency = np.zeros((81, 81))
        adjacency[:nb_splines][...,:nb_splines] = 1
        adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
        adjacency[:nb_splines][...,55:55+nb_agents] = 1
        adjacency[55:55+nb_agents][...,:nb_splines] = 1
        adj = torch.Tensor(adjacency).int().to(device)

        c_mask = torch.Tensor(adjacency[:,0]).int().to(device)

        timestamp = torch.tensor(moment/0.1).float().to(device)
        gtxy = torch.zeros((2*self.step+1)**2, 30, 2).int().to(device)
        
        return traj.unsqueeze(0), splines.unsqueeze(0), lanefeature.unsqueeze(0), adj.unsqueeze(0), \
    A_f.unsqueeze(0), c_mask.unsqueeze(0), timestamp.unsqueeze(0), gtxy.unsqueeze(0)
    
    def test_sampling_old(self, index, moment):
        traj = torch.tensor(self.T[index]).float().to(device)
        splines = torch.tensor(self.M[index]).float().to(device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(device)
        nb_agents = self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        a = self.Adj[index].toarray()
        af = a.copy()#+np.eye(55)
        af[af<0] = 0
        pad = np.zeros((55,55))
        pad[:nb_splines,:nb_splines]=np.eye(nb_splines)

        Af = np.linalg.matrix_power(af+pad+af.T, 4)
        Af[Af>0]=1

        A_f = torch.Tensor(Af).float().to(device)

        adjacency = np.zeros((81, 81))
        adjacency[:nb_splines][...,:nb_splines] = 1
        adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
        adjacency[:nb_splines][...,55:55+nb_agents] = 1
        adjacency[55:55+nb_agents][...,:nb_splines] = 1
        adj = torch.Tensor(adjacency).int().to(device)

        c_mask = torch.Tensor(adjacency[:,0]).int().to(device)
        
        return traj.unsqueeze(0), splines.unsqueeze(0), lanefeature.unsqueeze(0), adj.unsqueeze(0), \
    A_f.unsqueeze(0), c_mask.unsqueeze(0)


class InteractionDataset_inf(Dataset):
    """
    filename: a list of files or one filename of the .npz file
    stage: {"train", "val", "test"}
    """
    def __init__(self, filenames, stage, para, moment, selected, lrange):
        self.stage= stage
        self.para = para
        self.step = para['step']
        self.moment = moment
        self.resolution = para['resolution']
        self.sigmax = para['sigmax']
        self.sigmay = para['sigmay']
        self.lrange = lrange
        if stage == 'train':
            self.T = []
            self.M = []
            self.L = []
            self.N_agents = []
            self.N_splines = []
            self.Adj = []
            for filename in filenames:
                data = np.load('./interaction_merge/'+filename+'.npz', allow_pickle=True)

                self.T.append(data['trajectory'])
                self.M.append(data['maps'])
                self.L.append(data['lanefeature'])
                self.N_agents.append(data['nbagents'])
                self.N_splines.append(data['nbsplines'])
                self.Adj.append(data['adjacency'])

            self.T = np.concatenate(self.T, axis=0)[selected]
            self.M = np.concatenate(self.M, axis=0)[selected]
            self.L = np.concatenate(self.L, axis=0)[selected]
            self.N_agents = np.concatenate(self.N_agents, axis=0)[selected]
            self.N_splines = np.concatenate(self.N_splines, axis=0)[selected]
            self.Adj = np.concatenate(self.Adj, 0)[selected]
                
            # data = np.load('./interaction_merge/traj_train.npz', allow_pickle=True)
            # self.Y = data['traj']
        else:
            data = np.load('./interaction_merge/'+filenames[0]+'.npz', allow_pickle=True)
            self.T = data['trajectory']
            self.M = data['maps']
            self.L = data['lanefeature']
            self.N_agents = data['nbagents']
            self.N_splines = data['nbsplines']
            self.Adj = data['adjacency']

            if stage=='val':
                data = np.load('./interaction_merge/traj_val.npz', allow_pickle=True)
                self.Y = data['traj']
        
    def __len__(self):
        return len(self.N_agents)
        #return self.lrange[1] - self.lrange[0]
        
    def __getitem__(self, ind):
        #index = ind + self.lrange[0]
        index = ind
        traj = torch.tensor(self.T[index]).float().to(device)
        #Ts = self.T[index]
        #Ts[1:] = 0
        #traj = torch.tensor(Ts).float().to(device)
        splines = torch.tensor(self.M[index]).float().to(device)
        lanefeature = torch.tensor(self.L[index].toarray()).float().to(device)
        nb_agents = 1#self.N_agents[index]
        nb_splines = self.N_splines[index]
        
        a = self.Adj[index].toarray()
        af = a.copy()#+np.eye(55)
        af[af<0] = 0
        pad = np.zeros((55,55))
        pad[:nb_splines,:nb_splines]=np.eye(nb_splines)

        Af = np.linalg.matrix_power(af+pad+af.T, 4)
        Af[Af>0]=1

        A_f = torch.Tensor(Af).float().to(device)

        adjacency = np.zeros((81, 81))
        adjacency[:nb_splines][...,:nb_splines] = 1
        adjacency[55:55+nb_agents][...,55:55+nb_agents] = 1
        adjacency[:nb_splines][...,55:55+nb_agents] = 1
        adjacency[55:55+nb_agents][...,:nb_splines] = 1
        adj = torch.Tensor(adjacency).int().to(device)

        c_mask = torch.Tensor(adjacency[:,0]).int().to(device)

        timestamp = torch.tensor(self.moment).float().to(device)
        gtxy = torch.zeros((2*self.step+1)**2, 30, 2).int().to(device)
        
        return traj, splines, lanefeature, adj, A_f, c_mask, timestamp, gtxy