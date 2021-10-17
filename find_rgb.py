from numpy.lib.function_base import diff
import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp, time 
import scipy.interpolate
import open3d as o3d
import logging, sys, os
from io import StringIO as StringBuffer 
import plyfile



worgbpath = 'all_files_labeled_and_with_nolabel_points_worgb'
meshpath = 'meshes'
outpath = 'test'
valwith = []
valwo = []

fileswo = sorted(glob.glob(os.path.join(worgbpath,'scene*.pth')))
meshfiles = sorted(glob.glob(os.path.join(meshpath,'scene*.ply')))



for x in torch.utils.data.DataLoader(fileswo,collate_fn=lambda x: torch.load(x[0])):
    valwo.append(x) 

for i in range(len(valwo)): 
    file_name = os.path.split(fileswo[i])[-1] 
    coords,labels=valwo[i] 
    meshfile = meshfiles[i] 
    # print(file_name[:12]) # scene0011_00
    assert os.path.split(meshfile)[-1][:12] == file_name[:12] # 'scene0011_00_vh_clean_2.ply'
    data = plyfile.PlyData().read(meshfile)
    v=np.array([list(x) for x in data.elements[0]])
    coordsfrommesh=torch.from_numpy(np.ascontiguousarray(v[:,:3]-v[:,:3].mean(0)))
    colorsfrommesh=torch.from_numpy(np.ascontiguousarray(v[:,3:6])/127.5-1).double()
    coords = torch.tensor(coords)
    

    # assert torch.all(coords==coordsfrommesh)
    assert torch.all(torch.abs(coords-coordsfrommesh) < 1e-7 )
    torch.save((coords,colorsfrommesh,labels),os.path.join(outpath,file_name))