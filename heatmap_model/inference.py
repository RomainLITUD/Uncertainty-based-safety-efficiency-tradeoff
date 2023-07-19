from heatmap_model.utils import *
from heatmap_model.uncertainty_utils import *
from heatmap_model.interaction_dataset import *
import pickle
import pandas as pd
import time
import gc
from tqdm import tqdm

# def rotation_matrix(rad):
#     psi = math.pi/2-rad
#     return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])

def Inference_Polygon_train(model, para, selected):

    lrange = [0,0]
    testset = InteractionDataset_inf(['train1', 'train2','train3','train4'], 'train', para, moment=30, selected=selected, lrange=lrange)
    Hi = InferenceModel_train(model, testset, para)
    Ps = get_polygons_single(Hi)
    return Ps

def Inference_Polygon(model, para, dataname, nmax=107848, T=30):

    Ps = []

    slices = [s for s in range(0, 100009, 5000)] + [nmax]

    for start in range(20, len(slices)-1):
        lrange = [slices[start], slices[start+1]]
        H = []
        for i in range(1,T+1):
            print(start, i, end='\r')
            testset = InteractionDataset_inf([dataname], 'val', para, moment=i, lrange=lrange)
            Hi = InferenceModel_single(model, testset, para)
            H.append(Hi)
        H = np.array(H)
        pl = get_polygons(H)
        Ps = Ps + pl
    return Ps

def Inference_Polygon_sup(model, para, dataname, nmax=107848, T=30):

    Ps = []

    lrange = [5000, 5000+7848]
    H = []
    for i in range(1,T+1):
        print(i)
        testset = InteractionDataset_inf([dataname], 'val', para, moment=i, lrange=lrange)
        Hi = InferenceModel_single(model, testset, para)
        H.append(Hi)
    H = np.array(H)
    pl = get_polygons(H)
    Ps = Ps + pl
    return Ps


def InferenceModel(model, para, dataname, T=30):
    H = []
    
    for i in range(1,T+1):
        print(i, end='\n')
        testset = InteractionDataset_inf([dataname], dataname, para, moment=i)
        Hi = InferenceModel_single(model, testset, para)
        H.append(Hi)
    
    return np.array(H)

def InferenceModel_single(model, dataset, para):
    H = []  
    
    nb = len(dataset)
    cut = list(range(0, nb, 400*6)) + [nb]
    
    for i in range(len(cut)-1):
        print(i)
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=6, shuffle=False)
        
        for k, data in enumerate(loader):
            print(k, end='\r')
            traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy = data
            heatmap = model(traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy)
            H.append(heatmap.detach().to('cpu').numpy())   
    H = np.concatenate(H, 0).transpose((1,2,0))
    H = np.transpose(H/np.amax(H, (0,1)), (2,0,1))
    H[H<0.05] = 0
    H = np.array([csr_matrix(h) for h in H])
    return H

def InferenceModel_train(model, dataset, para):
    H = []  
    
    nb = len(dataset)
    cut = list(range(0, nb, 400*6)) + [nb]
    
    for i in range(len(cut)-1):
        print(i)
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=6, shuffle=False)
        Hi = []
        for k, data in enumerate(loader):
            print(k, end='\r')
            traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy = data
            heatmap = model(traj, splines, lanefeature, adj, af, c_mask, timestamp, gtxy)
            Hi.append(heatmap.detach().to('cpu').numpy())
        Hi = np.concatenate(Hi, 0).transpose((1,2,0))
        Hi = np.transpose(Hi/np.amax(Hi, (0,1)), (2,0,1))   
        Hi[Hi<0.05] = 0
        Hi = np.array([csr_matrix(h) for h in Hi])

        H.append(Hi)
    H = np.concatenate(H, 0)
    return H

def InferenceModel_noe(model, dataset, para, k=6):
    Yp = []
    
    nb = len(dataset)
    print(nb)
    cut = list(range(0, nb, 400*4)) + [nb]
    
    for i in range(len(cut)-1):
        print(i, end='\r')
        ind = list(range(cut[i], cut[i+1]))
        testset = torch.utils.data.Subset(dataset, ind)
        loader = DataLoader(testset, batch_size=4, shuffle=False)
        
        H = []
        for k, data in enumerate(loader):
            traj, maps, lanefeatures, adj, Af, c_mask, timestamp, gtxy = data
            #print(traj.size())
            heatmap = model(traj, maps, lanefeatures, adj, Af,c_mask, timestamp, gtxy)
            H.append(heatmap.detach().to('cpu').numpy())
        H = np.concatenate(H,0)
        H = np.transpose(H, (1,2,0))
        H = (H/np.sum(H, axis=(0,1))).transpose((2,0,1))

        yp = ModalSampling_old(H, para, r=3, k=6)
        Yp.append(yp)
        
    Yp = np.concatenate(Yp, 0)        
    
    return Yp
        
def Generate_csv(trajmodel, filename, Yp):
    print('loading model and data...')
    F = Yp.reshape(-1, 2)
    testset = InferenceTraj(F)
    loader = DataLoader(testset, batch_size=16, shuffle=False)
    print(len(testset))
    data = np.load('./interaction_merge/test.npz', allow_pickle=True)
    translate = data['origin']
    R = data['radian']
    nb = len(R)
    
    frame_ = np.arange(11,41)
    
    rotate = np.array([rotation_matrix(theta) for theta in R])
    
    with open('./interaction_merge/testfile.pickle', 'rb') as f:
        testfile = pickle.load(f)
    
    with open('./interaction_merge/test_index.pickle', 'rb') as f:
        Dnew = pickle.load(f)
    samplelist = Dnew[0]
    tracklist = Dnew[1]
    
    file_id = [int(case[:-6])-1 for case in samplelist]
    case_id = [int(case[-6:]) for case in samplelist]
    track_id = [int(track) for track in tracklist]
        
    trajmodel.load_state_dict(torch.load(filename+'traj.pt'))
    
    print('Completing trajectories...')
    T = []
    for k, x in enumerate(loader):
        print(16*k, end='\r')
        traj = trajmodel(x)
        T.append(traj.detach().to('cpu').numpy())
    T = np.concatenate(T, 0).reshape(-1, 6, 29, 2)   
    T = np.concatenate([T, np.expand_dims(Yp, 2)], -2) #(N, 6, 30, 2)
    
    T = np.einsum('bknf,bfc->bknc', T, rotate)
    T = np.transpose(T, (1,2,0,3))
    T = np.transpose((T+translate), (2,0,1,3))
    print(T.shape)
    print('generating submission logs...')
    
    for i in range(17):
        print(i,'th file...', end='\r')
        D = {}
        indices = [pos for pos in range(nb) if file_id[pos]==i]
        case = np.array([case_id[index] for index in indices])
        track = np.array([track_id[index] for index in indices])
        traj = T[indices]
        
        nb_case = len(indices)
        case = list(np.repeat(case, 30))
        track = list(np.repeat(track, 30))
        #print(indices)
        
        frame = np.tile(frame_, nb_case)
        
        D['case_id'] = case
        D['track_id'] = track
        D['frame_id'] = frame
        D['timestamp_ms'] = (100*frame).tolist()
        for k in range(1,7):
            D['x'+str(k)] = traj[:,k-1,:,0].flatten().tolist()
            D['y'+str(k)] = traj[:,k-1,:,1].flatten().tolist()
            
        df = pd.DataFrame(D)
        df.sort_values(by=['case_id'])
        
        subfile = './submission/'+testfile[i][:-7]+'sub.csv'
        
        df.to_csv(subfile,index=False)