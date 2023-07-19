import numpy as np
import pickle
import os
import math
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from shapely.geometry import Polygon, LineString, Point

import xml.etree.ElementTree as xml
import pyproj
import sys
import pandas as pd
from . import map_vis_without_lanelet

def get_polygon_cars(center, width, length, radian):
    x0, y0 = center[0], center[1]
#     lowleft = (x0 - length / 2., y0 - width / 2.)
#     lowright = (x0 + length / 2., y0 - width / 2.)
#     upright = (x0 + length / 2., y0 + width / 2.)
#     upleft = (x0 - length / 2., y0 + width / 2.)
    lowleft = (- length / 2., - width / 2.)
    lowright = ( + length / 2., - width / 2.)
    upright = ( + length / 2., + width / 2.)
    upleft = ( - length / 2., + width / 2.)
    rotate_ = rotation_matrix(radian)
    
    return (np.array([lowleft, lowright, upright, upleft])).dot(rotate_)+center

def get_polygon_peds(center, width, length):
    x0, y0 = center[0], center[1]
    lowleft = (x0 - length / 2., y0 - width / 2.)
    lowright = (x0 + length / 2., y0 - width / 2.)
    upright = (x0 + length / 2., y0 + width / 2.)
    upleft = (x0 - length / 2., y0 + width / 2.)
    
    return np.array([lowleft, lowright, upright, upleft])


def rotation_matrix(rad):
    psi = rad - math.pi/2
    return np.array([[math.cos(psi), -math.sin(psi)],[math.sin(psi), math.cos(psi)]])


def Visualize(H, Y, Ys, origin, radian, polygons, filename, case_id, track_id, title, xrange, yrange, para, frame_id=10):
    resolution = para['resolution']
    ymax = para['ymax']
    
    rotate = rotation_matrix(radian)
    mapname = './maps/'+filename[:-8]+'.osm'
    if "obs" in filename:
        df = pd.read_csv('./rawdata/test/'+filename)
    else:
        df = pd.read_csv('./rawdata/val/'+filename)
    df = df.query('case_id=='+str(case_id))
    df_e = df[df['frame_id']<=frame_id]
    
    all_agents = set(df_e['track_id'].values)
    
    # Visualization module
    fig, axes = plt.subplots(1, 1, figsize=(8.5, 12))
    fig.canvas.set_window_title(title)
    map_vis_without_lanelet.draw_map_without_lanelet(mapname, axes, origin, rotate)

    for polygon in polygons:
        if len(polygon) > 3:
            p = np.array(polygon)
            px = p[:,1]/2-23
            py = p[:,0]/2-12
            p = Polygon(np.array([px, py]).T)
            axes.plot(*p.exterior.xy)
    
    for ind in all_agents:
        dfc = df_e[df_e['track_id']==ind]
        agent_type = dfc['agent_type'].values[0]
        traj_obs = np.stack([dfc['x'].values, dfc['y'].values], -1)
        traj_obs = (traj_obs-origin).dot(rotate)
        width = dfc['width'].values[-1]
        length = dfc['length'].values[-1]
        center = traj_obs[-1]
        
        if agent_type=='car':
            type_dict = dict(color="green", linewidth=3, zorder=11)
            yaw = radian-dfc['psi_rad'].values[-1]
#             if center[0]==0:
#                 length = 12
#                 width = 2.85
            bbox = get_polygon_cars(center, width, length, yaw)
            if center[0]==0:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='red', zorder=20, edgecolor='red', linewidth=1, alpha=0.5)
            else:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='blue', zorder=20, edgecolor='blue', linewidth=1, alpha=0.5)
            axes.add_patch(rect)
        else:
            type_dict = dict(color="pink", linewidth=3, zorder=11)
            bbox = get_polygon_peds(center, width, length)
            rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='none', zorder=20, edgecolor='pink')
            axes.add_patch(rect)
        
        axes.plot(traj_obs[:,0], traj_obs[:,1], **type_dict)
    x = np.arange(-23, 23+resolution, resolution)
    y = np.arange(-12, ymax+resolution, resolution)
    s = H.copy()
    s[s<0.05]=np.nan
    axes.pcolormesh(x, y, s.transpose(), cmap='Reds',zorder=0)
    for i in range(len(Y)):
        axes.scatter(Y[i,0], Y[i,1], s=70, alpha=0.7, zorder=199, marker='*', 
                   facecolors='none', edgecolors='black', linewidth=1)
    for i in range(len(Ys)):
        if i==0:
            axes.scatter(Ys[i,0], Ys[i,1], s=200, zorder=200, marker='*', 
                   facecolors='black', edgecolors='black', linewidth=1, label='1.5s')
        else:
            axes.scatter(Ys[i,0], Ys[i,1], s=70, zorder=200, marker='*', 
                       facecolors='black', edgecolors='black', linewidth=1)
    axes.set_title(title, fontsize=15)
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
    if list(Ys):
        axes.legend(fontsize=14)
    
    return fig, axes

def Visualize_index(index, H, Y, Ys, polygons, title, xrange, yrange, para, mode='test'):
    if mode=='test':
        datafiles = os.listdir('./rawdata/test/')
        datafiles.sort()
        data = np.load('./interaction_merge/test.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/test_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)
    if mode == 'val':
        datafiles = os.listdir('./rawdata/val/')
        datafiles.sort()
        data = np.load('./interaction_merge/vis_val.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/val_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)

    if mode == 'valall':
        datafiles = os.listdir('./rawdata/val/')
        datafiles.sort()
        data = np.load('./interaction_merge/vis_valall.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/valall_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)
    samplelist = Dnew[0]
    tracklist = Dnew[1]

    file_id = int(samplelist[index][:-6])-1
    case_id = int(samplelist[index][-6:])
    track_id = int(tracklist[index])
    
    return Visualize(H, Y, Ys, origin, radian, polygons, datafiles[file_id], case_id, track_id, title, xrange, yrange, para, frame_id=10)

def Visualize_double(H, Y, Ys, origin, radian, filename, case_id, track_id, title, xrange, yrange, para, fig, axes, frame_id=10):
    resolution = para['resolution']
    ymax = para['ymax']
    
    rotate = rotation_matrix(radian)
    mapname = './maps/'+filename[:-8]+'.osm'
    if "obs" in filename:
        df = pd.read_csv('./rawdata/test/'+filename)
    else:
        df = pd.read_csv('./rawdata/val/'+filename)
    df = df.query('case_id=='+str(case_id))
    df_e = df[df['frame_id']<=frame_id]
    
    all_agents = set(df_e['track_id'].values)
    
    # Visualization module
    #fig, axes = plt.subplots(1, 1, figsize=(8.5, 8.5))
    fig.canvas.set_window_title(title)
    map_vis_without_lanelet.draw_map_without_lanelet(mapname, axes, origin, rotate)
    
    for ind in all_agents:
        dfc = df_e[df_e['track_id']==ind]
        agent_type = dfc['agent_type'].values[0]
        traj_obs = np.stack([dfc['x'].values, dfc['y'].values], -1)
        traj_obs = (traj_obs-origin).dot(rotate)
        width = dfc['width'].values[-1]
        length = dfc['length'].values[-1]
        center = traj_obs[-1]
        
        if agent_type=='car':
            type_dict = dict(color="green", linewidth=3, zorder=11)
            yaw = radian-dfc['psi_rad'].values[-1]
#             if center[0]==0:
#                 length = 12
#                 width = 2.85
            bbox = get_polygon_cars(center, width, length, yaw)
            if center[0]==0:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='red', zorder=20, edgecolor='red', linewidth=1, alpha=0.5)
            else:
                rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='blue', zorder=20, edgecolor='blue', linewidth=1, alpha=0.5)
            axes.add_patch(rect)
        else:
            type_dict = dict(color="pink", linewidth=3, zorder=11)
            bbox = get_polygon_peds(center, width, length)
            rect = matplotlib.patches.Polygon(bbox, closed=True, facecolor='none', zorder=20, edgecolor='pink')
            axes.add_patch(rect)
        
        axes.plot(traj_obs[:,0], traj_obs[:,1], **type_dict)
    x = np.arange(-23, 23+resolution, resolution)
    y = np.arange(-12, ymax+resolution, resolution)
    s = H.copy()
    s[s<0.05]=np.nan
    axes.pcolormesh(x, y, s.transpose(), cmap='Reds',zorder=0)
    for i in range(len(Y)):
        axes.scatter(Y[i,0], Y[i,1], s=70, alpha=0.7, zorder=199, marker='*', 
                   facecolors='none', edgecolors='black', linewidth=1)
    for i in range(len(Ys)):
        if i==0:
            axes.scatter(Ys[i,0], Ys[i,1], s=70, zorder=200, marker='*', 
                   facecolors='black', edgecolors='black', linewidth=1, label='Final Position')
        else:
            axes.scatter(Ys[i,0], Ys[i,1], s=70, zorder=200, marker='*', 
                       facecolors='black', edgecolors='black', linewidth=1)
    axes.set_title(title, fontsize=15)
    axes.set_xlim(xrange[0], xrange[1])
    axes.set_ylim(yrange[0], yrange[1])
#     if list(Ys):
#         axes.legend(fontsize=12, loc='best')
    
    return fig, axes

def Visualize_index_double(index, H, Y, Ys, title, xrange, yrange, para, fig, axes, mode='test'):
    if mode=='test':
        datafiles = os.listdir('./rawdata/test/')
        datafiles.sort()
        data = np.load('./interaction_merge/test.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/test_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)
    if mode == 'val':
        datafiles = os.listdir('./rawdata/val/')
        datafiles.sort()
        data = np.load('./interaction_merge/vis_val.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/val_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)

    if mode == 'valall':
        datafiles = os.listdir('./rawdata/val/')
        datafiles.sort()
        data = np.load('./interaction_merge/vis_valall.npz', allow_pickle=True)
        origin = data['origin'][index]
        radian = data['radian'][index]
        with open('./interaction_merge/valall_index.pickle', 'rb') as f:
            Dnew = pickle.load(f)
    samplelist = Dnew[0]
    tracklist = Dnew[1]

    file_id = int(samplelist[index][:-6])-1
    case_id = int(samplelist[index][-6:])
    track_id = int(tracklist[index])
    
    return Visualize_double(H, Y, Ys, origin, radian, datafiles[file_id], case_id, track_id, title, xrange, yrange, para, fig, axes, frame_id=10)