from tqdm import tqdm
import numpy as np
import math
from scipy.signal import convolve2d
from imantics import Polygons, Mask
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from skimage.transform import rescale
import shapely.affinity
from shapely.geometry import box
import pickle
import gc
import os

def rotation_matrix_back(rad):
    psi = rad - math.pi/2
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])

def read_transform():
    dt = np.load('interaction_merge/vis_valall.npz')
    R1 = dt['radian']
    C1 = dt['origin']
    dt = np.load('interaction_merge/test.npz')
    R2 = dt['radian']
    C2 = dt['origin']

    return np.concatenate([C1, C2], 0), np.concatenate([R1, R2])

def selected_trainset():
    datafiles_train = os.listdir('./rawdata/train/')
    datafiles_train.sort()
    with open('./interaction_merge/train_index.pickle', 'rb') as f:
        Dnew = pickle.load(f)
    samplelist = Dnew[0]
    tracklist = Dnew[1]

    #candidate = ['DR_DEU_Roundabout_OF', 'DR_USA_Intersection_GL', 'DR_USA_Intersection_MA', 'DR_USA_Roundabout_FT']
    #order_c = [4, 13, 14, 16]
    candidate = ['DR_USA_Intersection_GL', 'DR_USA_Roundabout_FT']
    order_c = [13, 16]

    count = 0

    selected = []
    cases = []

    for i in tqdm(range(len(samplelist))):
        fileid = int(samplelist[i][:-6])-1
        mapname = datafiles_train[fileid][:-10]
        if mapname in candidate:
            ind = candidate.index(mapname)
            gc.enable()
            for j in tracklist[i]:
                selected.append(count)
                cases.append(order_c[ind])
                count += 1
            gc.disable()
        else:
            for j in tracklist[i]:
                count += 1
    return selected, cases

def read_map_order():
    datafiles_val = os.listdir('./rawdata/val/')
    datafiles_val.sort()
    with open('./interaction_merge/valall_index.pickle', 'rb') as f:
        Dnew = pickle.load(f)
    samplelist_val = Dnew[0]

    datafiles_test = os.listdir('./rawdata/test/')
    datafiles_test.sort()
    with open('./interaction_merge/test_index.pickle', 'rb') as f:
        Dnew = pickle.load(f)
    samplelist_test = Dnew[0]

    with open('./poly_results/geometry_all.pkl', 'rb') as f:
        Maps = pickle.load(f)
    map_order = list(Maps.keys())

    Mval = np.zeros(len(samplelist_val)).astype(int)
    Mtest = np.zeros(len(samplelist_test)).astype(int)

    for i in range(len(samplelist_val)):
        fileid = int(samplelist_val[i][:-6])-1
        mapname = datafiles_val[fileid][:-8]+'.osm'
        Mval[i] = map_order.index(mapname)

    for i in range(len(samplelist_test)):
        fileid = int(samplelist_test[i][:-6])-1
        mapname = datafiles_test[fileid][:-8]+'.osm'
        Mtest[i] = map_order.index(mapname)

    Mall = np.concatenate([Mval, Mtest])
    return Mall, Maps

def get_polygons_single(H, thred=0.1):
    N = len(H)
    POLYGONS_LIST = []
    for i in tqdm(range(N)):
        Hr = H[i].toarray()
        Hr[Hr<thred]=0
        Hr[Hr>=thred]=1
        polygons = Mask(Hr).polygons()
        polys = [po.tolist() for po in polygons.points]
        gc.disable()
        POLYGONS_LIST.append(polys)
        gc.enable()
    return POLYGONS_LIST

def get_polygons(H, thred=0.1):
    N = len(H[0])
    POLYGONS_LIST = []
    for i in tqdm(range(N)):
        POLYGONS = []
        for j in range(30):
            Hr = H[j,i].toarray()
            Hr[Hr<thred]=0
            Hr[Hr>=thred]=1
            polygons = Mask(Hr).polygons()
            polys = [po.tolist() for po in polygons.points]
            POLYGONS.append(polys)
        POLYGONS_LIST.append(POLYGONS)
    return POLYGONS_LIST

def construct_conv_kernel(r, resolution):
    x = np.linspace(-r, r, int(2*r/resolution)+1)
    y = np.linspace(-r, r, int(2*r/resolution)+1)
    xv, yv = np.meshgrid(x, y)
    dist = xv**2+yv**2
    kernel = np.where(dist < r**2, np.ones_like(dist), np.zeros_like(dist))

    return kernel

def confidence_horizon_conv(H, V, dx, thred=0.1, buffer=2.):
    driving_space = np.maximum(V*6, 17.5)

    N = len(H[0])
    A = np.zeros((N, 30))
    ch = np.zeros(N)

    for i in tqdm(range(N)):
        for j in range(30):
            A[i, j] = occupied_space(H[j,i].toarray(), thred, buffer, dx)

    for i in range(N):
        indices = np.where(A[i]>driving_space[i])[0].tolist()
        if indices:
            ch[i] = (indices[0]+1)/10
        else:
            ch[i] = 3

    return A, ch

def occupied_space(H, thred, buffer, dx):
    kernel = construct_conv_kernel(buffer, dx)

    H[H<thred] = 0.
    H[H>=thred] = 1.

    Hr = convolve2d(H, kernel, mode='same', boundary='fill', fillvalue=0)

    area = len(np.where(Hr.flatten()>0)[0])*dx*dx

    return area

def get_occupied_area(H, Ht, safety_buffer, dx=0.5, thred=0.1):
    ratio = 1/dx/dx
    N = len(H)
    A = np.zeros((N, 30))

    for i in tqdm(range(N)):
        for j in range(30):
            shapes = H[i][j]
            try:
                if len(shapes) == 1:
                    p = Polygon(np.array(shapes[0]))
                    p_offset = Polygon(p.buffer(safety_buffer, resolution=16, cap_style=1, join_style=1).exterior)
                    A[i,j] = p_offset.area/ratio

                else:
                    p_list = [Polygon(np.array(shape)) for shape in shapes]
                    p_offset = [Polygon(p.buffer(safety_buffer, resolution=16, cap_style=1, join_style=1).exterior) for p in p_list]

                    merged_polygon = unary_union(p_offset)
                    A[i,j] = merged_polygon.area/ratio
            except ValueError:
                A[i,j] = occupied_space(Ht[j,i].toarray(), thred, safety_buffer//dx*dx, dx)

    return A

def get_occupied_area_speed(H, Ht, V, b, safety_buffer, dx=0.5, thred=0.1):
    ratio = 1/dx/dx
    N = len(H)
    A = np.zeros((N, 30))

    for i in tqdm(range(N)):
        for j in range(30):
            shapes = H[i][j]
            buffer = max(safety_buffer, b*V[i])
            try:
                if len(shapes) == 1:
                    p = Polygon(np.array(shapes[0]))
                    p_offset = Polygon(p.buffer(buffer, resolution=16, cap_style=1, join_style=1).exterior)
                    A[i,j] = p_offset.area/ratio

                else:
                    p_list = [Polygon(np.array(shape)) for shape in shapes]
                    p_offset = [Polygon(p.buffer(buffer, resolution=16, cap_style=1, join_style=1).exterior) for p in p_list]

                    merged_polygon = unary_union(p_offset)
                    A[i,j] = merged_polygon.area/ratio
            except ValueError:
                A[i,j] = occupied_space(Ht[j,i].toarray(), thred, buffer//dx*dx, dx)

    return A

def get_confidence_horizon(A, V, k=6, b=17.5):
    driving_space = np.maximum(V*k, b)
    N = len(V)
    ch = np.zeros(N)

    for i in range(N):
        indices = np.where(A[i]>driving_space[i])[0].tolist()
        if indices:
            ch[i] = (indices[0]+1)/10
        else:
            ch[i] = 3

    return ch

def get_nb_mode(H, thred=0.1):
    N = len(H[0])
    M = np.zeros((N, 30))
    for i in tqdm(range(N)):
        for j in range(30):
            Hr = H[j,i].toarray()
            Hr[Hr<thred]=0
            Hr[Hr>=thred]=1
            polygons = Mask(Hr).polygons()
            M[i,j] = len(polygons.points)
    return M.astype(int)

def get_occupied_area_test(H, V, k, safety_buffer, dx=0.5, thred=0.1):
    ratio = 1/dx/dx
    N = len(H)
    A = np.zeros((N, 30))
    lateral = 3.25

    for i in tqdm(range(N)):
        for j in range(30):
            shapes = H[i][j]
            buffer = k*V[i]+safety_buffer

            p_list = []
            for shape in shapes:
                if len(shape) > 2:
                    boundary = np.array(shape)
                    x, y = boundary[:, 0], boundary[:, 1]
                    xmin, xmax = np.amin(x), np.amax(x)
                    ymin, ymax = np.amin(y), np.amax(y)

                    scalex = (xmax-xmin+2*buffer)/(xmax-xmin)
                    scaley = (ymax-ymin+2*lateral)/(ymax-ymin)

                    p = Polygon(boundary)
                    p_new = shapely.affinity.scale(p, scalex, scaley, origin ='centroid')
                    p_list.append(p_new)
                else:
                    boundary = np.array(shape)
                    cx, cy = np.mean(boundary[...,0]), np.mean(boundary[...,1])
                    circle = Point(cx, cy).buffer(1)
                    p_new = shapely.affinity.scale(circle, 2*buffer, 2*lateral, origin='centroid')

                    p_list.append(p_new)
            
            if len(p_list) > 1:
                merged_polygon = unary_union(p_list)
                A[i,j] = merged_polygon.area/ratio
            else:
                A[i,j] = p_list[0].area/ratio

    return A

def get_occupied_area_test1(H, V, k, safety_buffer, dx=0.5, thred=0.1):
    ratio = 1/dx/dx
    N = len(H)
    A = np.zeros((N, 30))
    lateral = 3*2

    for i in tqdm(range(N)):
        for j in range(30):
            shapes = H[i][j]
            buffer = (k*V[i]+safety_buffer)*2

            p_list = []
            for shape in shapes:
                extend_polygons = []
                boundary = np.array(shape)

                if len(shape) > 2:
                    extend_polygons.append(Polygon(boundary))

                for pos in boundary:
                    p_i = box(pos[0]-buffer, pos[1]-lateral, pos[0]+buffer, pos[1]+lateral)
                    extend_polygons.append(p_i)
                    
                p_new = unary_union(extend_polygons)
                p_list.append(p_new)
            
            if len(p_list) > 1:
                merged_polygon = unary_union(p_list)
                A[i,j] = merged_polygon.area/ratio
            else:
                A[i,j] = p_list[0].area/ratio

    return A

def get_occupied_area_test2(H, V, k, safety_buffer, dx=0.5, thred=0.1):
    ratio = 1/dx/dx
    N = len(H)
    A = np.zeros((N, 30))
    lateral = 3.25*2

    for i in tqdm(range(N)):
        for j in range(30):
            shapes = H[i][j]
            buffer = (k*V[i]+safety_buffer)*2

            p_list = []
            for shape in shapes:
                extend_polygons = []
                boundary = np.array(shape)

                if len(shape) > 2:
                    extend_polygons.append(Polygon(boundary))

                for pos in boundary:
                    circle = Point(pos[0], pos[1]).buffer(1)
                    p_i = shapely.affinity.scale(circle, 2*buffer, 2*lateral, origin='centroid')
                    extend_polygons.append(p_i)
                    
                p_new = unary_union(extend_polygons)
                p_list.append(p_new)
            
            if len(p_list) > 1:
                merged_polygon = unary_union(p_list)
                A[i,j] = merged_polygon.area/ratio
            else:
                A[i,j] = p_list[0].area/ratio

    return A

def read_polygons():
    dz = np.load('./results/valspeed.npz', allow_pickle=True)
    Vval = dz['V']
    dz = np.load('./results/testspeed.npz', allow_pickle=True)
    Vtest = dz['V']

    with open('./poly_results/val_scenario.pkl', 'rb') as f:
        sval = pickle.load(f)
    with open('./poly_results/test_scenario.pkl', 'rb') as f:
        stest = pickle.load(f)

    scenario = np.array(sval + stest)
    speed = np.concatenate([Vval, Vtest])

    with open('./poly_results/polygons_all.pkl', 'rb') as f:
        polygons = pickle.load(f)
    
    with open('./Uresult/polygon1.pkl', 'rb') as f:
        poly1 = pickle.load(f)
    polygons = polygons[:-22498] + poly1[5000:10000] + polygons[-22498:]

    return scenario, speed, polygons

def calculate_area(polygons, dx=0.5, buffer=None):
    ratio = 1/dx/dx
    p_list = []
    for poly in polygons:
        if len(poly) > 8:
            p = Polygon(np.array(poly))
            if buffer:
                p = Polygon(p.buffer(buffer, resolution=16, cap_style=1, join_style=1).exterior)
            p_list.append(p)

        else:
            boundary = np.array(poly)
            cx, cy = np.mean(boundary[...,0]), np.mean(boundary[...,1])
            if buffer:
                circle = Point(cx, cy).buffer(buffer)
            else:
                circle = Point(cx, cy).buffer(0.2)

            p_list.append(circle)
    
    if len(p_list) > 1:
        merged_polygon = unary_union(p_list)
        return merged_polygon.area/ratio
    else:
        return p_list[0].area/ratio

def get_scenario_based_area(polygon_list, dx=0.5, buffer=None):
    A = np.zeros((len(polygon_list), 30))

    for i in tqdm(range(len(polygon_list))):
        for j in range(30):
            a = calculate_area(polygon_list[i][j], dx=dx, buffer=buffer)
            A[i,j] = a

    return A

def get_scenario_based_area_timestep(polygon_list, t, dx=0.5, buffer=None):
    A = np.zeros(len(polygon_list))

    for i in tqdm(range(len(polygon_list))):
        a = calculate_area(polygon_list[i][t-1], dx=dx, buffer=buffer)
        A[i] = a

    return A

def get_scenario_based_area_train(polygon_list, dx=0.5, buffer=None):
    A = np.zeros(len(polygon_list))

    for i in tqdm(range(len(polygon_list))):
        a = calculate_area(polygon_list[i], dx=dx, buffer=buffer)
        A[i] = a

    return A

def get_offset_area(P, v, scenario, nr, k, still_buffer, lateral_buffer, dx=0.5):
    ratio = 1/dx/dx
    #lateral = lateral_buffer*2

    area = np.zeros(30)

    for j in range(30):
        shapes = P[j]
        buffer = (k*v+still_buffer)*2

        p_list = []
        for i in range(len(shapes)):
            shape = shapes[i]
            if nr[j][i] > 0 and scenario>1:
                lateral = 6*2
            else:
                lateral = lateral_buffer*2
            if len(shape) > 8:
                boundary = np.array(shape)
                x, y = boundary[:, 0], boundary[:, 1]
                xmin, xmax = np.amin(x), np.amax(x)
                ymin, ymax = np.amin(y), np.amax(y)

                scalex = (xmax-xmin+2*buffer)/(xmax-xmin)
                scaley = (ymax-ymin+2*lateral)/(ymax-ymin)

                p = Polygon(boundary)
                p_new = shapely.affinity.scale(p, scalex, scaley, origin ='centroid')
                p_list.append(p_new)
            else:
                boundary = np.array(shape)
                cx, cy = np.mean(boundary[...,0]), np.mean(boundary[...,1])
                circle = Point(cx, cy).buffer(1)
                p_new = shapely.affinity.scale(circle, buffer, lateral, origin='centroid')

                p_list.append(p_new)
        
        if len(p_list) > 1:
            merged_polygon = unary_union(p_list)
            area[j] =  merged_polygon.area/ratio
        else:
            area[j] =  p_list[0].area/ratio
    return area
        
def get_scenario_based_offset_area(polygon_list, scenario, Nr, V, k, still_buffer=6, lateral_buffer=3.25, dx=0.5):
    A = np.zeros((len(polygon_list), 30))

    for i in tqdm(range(len(polygon_list))):
        area = get_offset_area(polygon_list[i], V[i], scenario[i], Nr[i], k, still_buffer, lateral_buffer, dx)
        A[i] = area
    return A

def map_based_conflict(polygons, lanes, intersection, R, center):

    conflict = []
    for polygon in polygons:
        p = np.array(polygon)
        if len(p) > 8:
            px = p[:,1]/2-23
            py = p[:,0]/2-12
            p = np.array([px, py]).T
            p = p.dot(np.linalg.inv(R)) + center
            p = Polygon(p)
        else:
            cx, cy = np.mean(p[...,1])/2-23, np.mean(p[...,0])/2-12
            p = Point(cx, cy).buffer(0.2)

        cross_id = [i for i in range(len(lanes)) if p.intersects(Polygon(lanes[i]))]
        if len(cross_id) == 0:
            conflict.append(0)
        else:
            if np.amax(np.array(intersection)[cross_id]) > 0:
                conflict.append(1)
            else:
                conflict.append(0)

    return conflict

def determine_conflict(polygons, order, MAPS, origin, radian):
    Nr = []
    gc.disable()
    for i in tqdm(range(len(order))):
        mapname = list(MAPS.keys())[order[i]]
        center = origin[i]
        angle = radian[i]
        R = rotation_matrix_back(angle)

        lanes = MAPS[mapname][0]
        intersection = MAPS[mapname][1]
        conflict = []
        for j in range(30):
            cf = map_based_conflict(polygons[i][j], lanes, intersection, R, center)
            conflict.append(cf)
        Nr.append(conflict)
        
    gc.enable()

    return Nr