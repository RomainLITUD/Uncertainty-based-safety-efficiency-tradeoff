import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
import argparse
from scipy.sparse import csr_matrix
from skimage.transform import rescale
from skimage.measure import block_reduce
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage, misc
from scipy.signal import convolve2d
from skimage.filters import gaussian
from scipy.special import comb
from scipy.signal import savgol_filter
from scipy.special import comb
from typing import List, Tuple

from heatmap_model.interaction_dataset import *

import math
from torch.optim.optimizer import Optimizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mask_softmax(x, test=None):

    if test == None:
        return F.softmax(x, dim=-1)
    else:
        shape = x.shape
        if test.dim() == 1:
            test = torch.repeat_interleave(
                test, repeats=shape[1], dim=0)
        else:
            test = test.reshape(-1)
        x = x.reshape(-1, shape[-1])
        for i, j in enumerate(x):
            j[int(test[i]):] = -1e5
        return F.softmax(x.reshape(shape), dim=-1)
    
# def Entropy(x, dx, epsilon=1e-6):
#     x = x/np.sum(x)/dx/dx
#     y = np.where(x>epsilon, -x*np.log(x)*dx*dx, 0)
#     return np.sum(y)

def Entropy(x, resolution, epsilon=1e-4):
    x = np.reshape(x,(len(x),-1))
    x = x/np.sum(x,axis=-1)[:,np.newaxis]/resolution/resolution
    y = np.where(x>epsilon, -x*np.log(x+1e-4)*resolution*resolution, 0)
    return np.sum(y,axis=-1)

def KLDivergence(x,y, resolution, epsilon=1e-5):
    x = np.reshape(x,(len(x),-1))
    y = np.reshape(y,(len(y),-1))
    z = np.where(((y>epsilon)&(x>epsilon)), x*np.log(x/y)*resolution*resolution, 0)
    return np.sum(z,axis=-1)

def KLDivergence_uni(x,y, resolution, epsilon=1e-5):
    x = x/np.sum(x)
    y = y/np.sum(y)
    z = np.where((x*y)>epsilon, x*np.log(x/y)*resolution*resolution, 0)
    return np.sum(z)

def seq_entropy(H, dx):
    N = len(H[0])
    E = np.zeros((N, 30))
    for i in range(30):
        print(i, end='\r')
        Hi = np.array([hi.toarray() for hi in H[i]])
        E[:,i] = Entropy(Hi, dx)
    return E

def final_prediction(model, testset, para, radius=3, k=6, mode='test'):
    if mode == 'val':
        S = testset.Y   
    N = testset.__len__()
    
    Y = np.zeros((N,k,2))
    for ind in range(N):
        print(ind, end='\r')
        traj, maps, lanefeatures, adj, Af, c_mask, timestamp, gtxy = testset.test_sampling(ind, 3.)
        heatmap = model(traj, maps, lanefeatures, adj, Af, c_mask, timestamp, gtxy)
        hr = heatmap.detach().to('cpu').numpy()
        #hr = rescale(hr, 2, anti_aliasing=True)
        #del traj, maps, lanefeatures, adj, Af, c_mask, timestamp, gtxy
        
        #yp = ModalSampling(hr, resolution, para, r=radius, k=k)
        #yp = ModalSampling_refine(hr, para, r=7, k=k)
        yp = ModalSampling_old(hr, para, r=radius, k=k)
        Y[ind] = yp
        
    return Y

def trajectory_generation(H, para):
    dx = para['resolution']
    T = np.zeros((31,2))
    for i in range(len(H)):
        xc, yc = np.unravel_index(H[i].argmax(), H[i].shape)
        T[i+1] = np.array([-para['xmax']+dx*xc, para['ymin']+dx*yc])
    return T

def rawtrajectory_generation(H, para):
    dx = para['resolution']
    T = np.zeros((len(H),2))
    for i in range(len(H)):
        xc, yc = np.unravel_index(H[i].argmax(), H[i].shape)
        T[i] = np.array([-para['xmax']+dx*xc, para['ymin']+dx*yc])
    return T

def construct_conv_kernel(resolution, r):
    x = np.linspace(-r+resolution, r-resolution, int(2*r/resolution)-1)
    y = np.linspace(-r+resolution, r-resolution, int(2*r/resolution)-1)
    xv, yv = np.meshgrid(x, y)
    dist = xv**2+yv**2
    kernel = np.where(dist<4, np.ones_like(dist), np.zeros_like(dist))
    
    return kernel 

def pool2d_np(A, kernel_size, stride=1, padding=0):

    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    return A_w.max(axis=(2, 3))

def ModalSampling(H, resolution, para, r=2, k=6):
    kernel = construct_conv_kernel(resolution, r)
    bound = len(kernel)//2
    
    lx, ly = H.shape
    
    Y = np.zeros((k, 2))
    Hp = H.copy()
    xc, yc = np.unravel_index(Hp.argmax(), Hp.shape)
    xc = xc + bound
    yc = yc + bound
    Y[0] = np.array([-para['xmax']+resolution*xc, para['ymin']+resolution*yc])
    
    mask = np.logical_not(kernel)
    
    xlow = max(0, xc-bound)
    xhigh = min(lx, xc+bound+1)
    
    ylow = max(0, yc-bound)
    yhigh = min(ly, yc+bound+1)
    
    #print(ylow, yhigh, yc)
    
    Hp[xlow:xhigh, ylow:yhigh] = Hp[xlow:xhigh, ylow:yhigh]*mask[bound+xlow-xc:xhigh-xc+bound, bound+ylow-yc:yhigh-yc+bound]
    
    if k > 1:
        for j in range(1,k):
            Hr = convolve2d(Hp, kernel, mode='same', boundary='fill', fillvalue=0)
            xc, yc = np.unravel_index(Hr.argmax(), Hr.shape)
            Y[j] = np.array([-para['xmax']+resolution*xc, para['ymin']+resolution*yc])
            xlow = max(0, xc-bound)
            xhigh = min(lx, xc+bound+1)

            ylow = max(0, yc-bound)
            yhigh = min(ly, yc+bound+1)

            Hp[xlow:xhigh, ylow:yhigh] = Hp[xlow:xhigh, ylow:yhigh]*mask[bound+xlow-xc:xhigh-xc+bound, bound+ylow-yc:yhigh-yc+bound]

    return Y

def ModalSamplingm2(H, resolution, para, r=2, k=6):
    kernel = construct_conv_kernel(resolution, r)
    bound = len(kernel)//2
    
    lx, ly = H.shape
    
    Y = np.zeros((k, 2))
    Hp = H.copy()
    xc, yc = np.unravel_index(Hp.argmax(), Hp.shape)
    xc = xc
    yc = yc
    Y[0] = np.array([-para['xmax']+resolution*xc, para['ymin']+resolution*yc])
    Y[-1] = np.array([Y[0,0]+2, Y[0,1]])
    
    mask = np.logical_not(kernel)
    
    xlow = max(0, xc-bound)
    xhigh = min(lx, xc+bound+1)
    
    ylow = max(0, yc-bound)
    yhigh = min(ly, yc+bound+1)
    
    #print(ylow, yhigh, yc)
    
    Hp[xlow:xhigh, ylow:yhigh] = Hp[xlow:xhigh, ylow:yhigh]*mask[bound+xlow-xc:xhigh-xc+bound, bound+ylow-yc:yhigh-yc+bound]
    
    if k > 1:
        for j in range(1,k-1):
            Hr = convolve2d(Hp, kernel, mode='same', boundary='fill', fillvalue=0)
            xc, yc = np.unravel_index(Hr.argmax(), Hr.shape)
            Y[j] = np.array([-para['xmax']+resolution*xc, para['ymin']+resolution*yc])
            xlow = max(0, xc-bound)
            xhigh = min(lx, xc+bound+1)

            ylow = max(0, yc-bound)
            yhigh = min(ly, yc+bound+1)

            Hp[xlow:xhigh, ylow:yhigh] = Hp[xlow:xhigh, ylow:yhigh]*mask[bound+xlow-xc:xhigh-xc+bound, bound+ylow-yc:yhigh-yc+bound]

    return Y

def ModalSampling_old(H, paralist, r=2, k=6):
    # Hp = (N, H, W)
    dx, dy = paralist['resolution'], paralist['resolution']
    xmax, ymax = paralist['xmax'], paralist['ymax']
    ymin = paralist['ymin']
    Y = np.zeros((k, 2))

    xc, yc = np.unravel_index(H.argmax(), H.shape)
    xc=xc+r
    yc=yc+r
    pred = [-xmax+xc*dx, ymin+yc*dy]
    Y[0] = np.array(pred)
    
    H[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
    for j in range(1,k):
        Hr = pool2d_np(H, kernel_size=2*r+1, stride=1, padding=r)
        xc, yc = np.unravel_index(Hr.argmax(), Hr.shape)
        xc=xc+r
        yc=yc+r
        pred = [-xmax+xc*dx, ymin+yc*dy]
        Y[j] = np.array(pred)
        H[xc-r:xc+r+1,yc-r:yc+r+1] = 0.
    return Y

def ComputeError(Yp,Y, r=2, sh=6):
    assert sh <= Yp.shape[1]
    # Yp = [N,k,2], Y = [N,2]
    E = np.abs(Yp.transpose((1,0,2))-Y) #(k,N,2)
    FDE = np.min(np.sqrt(E[:sh,:,0]**2+E[:sh,:,1]**2), axis=0) #(N,)
    MR = np.where(FDE>r, np.ones_like(FDE), np.zeros_like(FDE))
    print("minFDE:", np.mean(FDE),"m")
    print("minMR:", np.mean(MR)*100,"%")
    return FDE, MR

def seq_area(H, dx, thred=0.1):
    N = len(H[0])
    E = np.zeros((N, 30))
    for i in range(30):
        print(i, end='\r')
        E[:,i] = Hi = np.array([(hi>=thred).sum() for hi in H[i]])*dx*dx
    return E

    

def NLLEstimate(H, Y, para):
    dx = para['resolution']
    H = H/np.sum(H)/dx/dx
    xmin, ymin = -para['xmax'], para['ymin']
    xc, yc = (Y[0]-xmin)//dx, (Y[1]-ymin)//dx
    return -np.log(H[int(xc), int(yc)]+1e-3)

def NLLEstimate_test(H, para):
    dx = para['resolution']
    H = H/np.sum(H)/dx/dx
    return -np.log(np.amax(H))

def bezier_curve(points, n=100):

    b = np.zeros((len(points), n))
    for i in range(len(points)):
        for j in range(n):
            b[i][j] = comb(len(points)-1, i) * (j/n)**i * (1-j/n)**(len(points)-1-i)

    curve = []
    for j in range(n):
        x, y = 0, 0
        for i in range(len(points)):
            x += b[i][j] * points[i][0]
            y += b[i][j] * points[i][1]
        curve.append((x, y))

    return np.array(curve)


def rawtrajectory_single(model, dataset, para, batchsize):
    Yp = []  
    
    nb = len(dataset)
    cut = list(range(0, nb, 400*batchsize)) + [nb]
    
    for i in range(len(cut)-1):
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=batchsize, shuffle=False)
        
        for k, data in enumerate(loader):
            print(k, end='\r')
            traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy = data
            heatmap = model(traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy)
            del traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy
            hr = heatmap.detach().to('cpu').numpy()
            Yp.append(rawtrajectory_generation(hr, para))
    Yp = np.concatenate(Yp, 0)
    return Yp

def rawtrajectory(model, para, dataname, batchsize, T=30):
    H = []
    
    for i in range(1,T+1):
        print(i, "timestep")
        testset = InteractionDataset_inf([dataname], dataname, para, moment=i)
        Hi = rawtrajectory_single(model, testset, para, batchsize)
        H.append(Hi)
    
    return np.array(H)