# https://github.com/pqhieu/jsis3d/blob/master/loaders/s3dis.py
import os
import h5py
import numpy as np
np.random.seed(0)
from tqdm import tqdm


import os.path
import numpy as np

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.datasets import S3DIS
from torch_geometric.data import DataLoader


class S3DIS_clean_instance(InMemoryDataset):
    def __init__(self, root, training=True):
        self.root = root
        self.split = 'train.txt' if training else 'test.txt'
        self.flist = os.path.join(self.root, 'metadata', self.split)
        self.rooms = [line.strip() for line in open(self.flist)]
        # print(self.rooms)
        # self.rooms = self.rooms[:2]
        # Load all data into memory
        self.coords_temp = []
        self.points_temp = []
        self.labels_temp = []
        print('> Loading h5 files...')
        for fname in tqdm(self.rooms, ascii=True):
            # fin = h5py.File(os.path.join(self.root, 'h5', fname))
            fin = h5py.File(os.path.join(self.root, fname))
            self.coords_temp.append(fin['coords'][:])
            self.points_temp.append(fin['points'][:])
            self.labels_temp.append(fin['labels'][:])
            fin.close()
        self.coords = np.concatenate(self.coords_temp, axis=0)
        self.points = np.concatenate(self.points_temp, axis=0)
        self.labels = np.concatenate(self.labels_temp, axis=0)



training=True
datamaxnum = 1
S3DIS_ins = S3DIS_clean_instance(root='data_with_ins_label/',training=training) 
dataset2 = S3DIS(root='per60_0.018_DBSCANCluster/',train=True)
print('----length of dataset----\n', len(dataset2))
data_loader2 = DataLoader(dataset2, batch_size=1, shuffle=False)
acc_cnt, overall_cnt = 0,0
for iteration, curr_obj in tqdm(enumerate(data_loader2)):
    y2 = curr_obj.y
    x2 = curr_obj.x
    z2 = curr_obj.z

    assert S3DIS_ins.labels[iteration,:,0].shape[0] == y2.shape[0]

    acc_cnt += sum(torch.from_numpy(np.array(S3DIS_ins.labels[iteration,:,0]))==y2)
    overall_cnt += len(y2)

    if iteration%100 ==99:
        print((acc_cnt+0.0)/(overall_cnt+0.0))
print('overall',(acc_cnt+0.0)/(overall_cnt+0.0))