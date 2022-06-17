# https://github.com/pqhieu/jsis3d/blob/master/loaders/s3dis.py
import os
import sys
import h5py
import numpy as np
np.random.seed(0)
import torch.utils.data as data
from tqdm import tqdm
import os.path as osp
import shutil
import time
import random
import argparse


import os.path
import pickle 
import numpy as np

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip) 
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree

# # if ClusterByPartitionMethods=True, use the following packages
# # from https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/partition.py
# sys.path.append("./superpoint_graph/partition/cut-pursuit/build/src")
# sys.path.append("./superpoint_graph/partition/ply_c")
# sys.path.append("./superpoint_graph/partition")
# import libcp
# import libply_c
# from graphs import *
# from provider import *






class S3DIS_instance(InMemoryDataset):
    def __init__(self, root, training=True, pre_cluster_path=None, precent_NL=80):
        self.root = root 
        self.precent_NL = int(precent_NL) # noise setting 
        self.is_Asymmetry = False 
        self.with_Cluster = True 
        if self.is_Asymmetry:
            self.with_Cluster = False 
        self.ClusterByPartitionMethods,self.ClusterByDBSCAN,self.ClusterByGMM = False,True,False 
        
        if self.with_Cluster and self.ClusterByPartitionMethods:
            self.reg_strength = 0.05
            self.out_root = 'per' + str(self.precent_NL)+'_'+ str(self.reg_strength) + '_EuclideanCluster/'
        elif self.with_Cluster and self.ClusterByDBSCAN: 
            self.MaxNeighborDistance = 0.018 
            self.out_root = 'per' + str(self.precent_NL)+'_'+ str(self.MaxNeighborDistance) + '_DBSCANCluster/'
            if self.ClusterByGMM:
                self.out_root = 'per' + str(self.precent_NL) + '_strongGMMCluster/'
        elif self.with_Cluster:
            self.out_root = 'per' + str(self.precent_NL)+'_Cluster/'
        elif self.is_Asymmetry:
            self.out_root = 'per' + str(self.precent_NL)+ '_Asymmetry/'
        else:
            self.out_root = 'per' + str(self.precent_NL)+ '/'
        
        
        if pre_cluster_path is not None:
            print('copying files ...')
            for item in os.listdir(pre_cluster_path):
                s = os.path.join(pre_cluster_path, item)
                d = os.path.join(self.out_root, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            self.copy_pre_cluster = True 
        else:
            try:
                os.makedirs(self.out_root+'processed/')
                os.makedirs(self.out_root+'raw/')
            except:
                print('!!!!!!!!!!!!!!out path already exist!!!!!!!!!!!!!!!!')
            
        self.out_processed_paths = [self.out_root+'processed/train_6.pt',self.out_root+'processed/test_6.pt']
        self.log_file = self.out_root+'raw/room_filelist.txt' if training else self.out_root+'raw/room_filelist_test.txt'
        self.split = 'train.txt' if training else 'test.txt' 


        self.flist = os.path.join(self.root, 'metadata', self.split)
        self.rooms = [line.strip() for line in open(self.flist)] 
        # Load all data into memory
        self.coords_temp = []
        self.points_temp = []
        self.labels_temp = []
        print('> Loading h5 files...')
        for fname in tqdm(self.rooms, ascii=True): 
            fin = h5py.File(os.path.join(self.root, fname))
            self.coords_temp.append(fin['coords'][:])
            self.points_temp.append(fin['points'][:])
            self.labels_temp.append(fin['labels'][:])
            fin.close()
        self.coords = np.concatenate(self.coords_temp, axis=0)
        self.points = np.concatenate(self.points_temp, axis=0)
        self.labels = np.concatenate(self.labels_temp, axis=0)
        # Post-processing
        self.dataset_size = self.points.shape[0]
        self.num_points = self.points.shape[1]
        for i in range(self.dataset_size):
            self.labels[i,:,1] = np.unique(self.labels[i,:,1], False, True)[1]
        self.max_instances = np.amax(self.labels[:,:,1]) + 1
        self.class_num = np.amax(self.labels[:,:,0]) + 1
        

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        masks = np.zeros((self.num_points, self.max_instances), dtype=np.float32)
        masks[np.arange(self.num_points), self.labels[i,:,1]] = 1
        return {
            'coords': self.coords[i],
            'points': self.points[i],
            'labels': self.labels[i],
            'masks': masks,
            'size': np.unique(self.labels[i,:,1]).size
        }

    

    def rand_label(self, y):
        t = time.time()*1000
        while t>0:
            t-=2**32
        t+=2**32
        np.random.seed(int(t))
        randidx_n = np.random.randint(0, y.shape[0])
        randidx_m = np.random.randint(0, y.shape[0])
        while y[randidx_n][0]==y[randidx_m][0]: # skip samples with the same class
            randidx_m = np.random.randint(0, y.shape[0])

        label = y[randidx_n].clone()
        label2 = y[randidx_m].clone()
        return randidx_n,randidx_m,label,label2

    # from https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/partition.py
    def partition(self,xyz, rgb, labels,voxel_width=0.0,dataset='s3dis',k_nn_adj=10,k_nn_geof=45,lambda_edge_weight=1.0,reg_strength=0.1,d_se_max=0): 
        if dataset=='s3dis': 
            n_labels = 13
            xyz = xyz - xyz.mean(0) 
            if voxel_width > 0:
                xyz, rgb, labels, dump = libply_c.prune(xyz.astype('f4'), voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
            else:
                xyz,rgb,labels = (xyz).astype('f4'),(rgb*255.0).astype('uint8'),labels.numpy().astype('uint8') 
                xyz[:,2] = xyz[:,2]*1.25 
        elif dataset=='custom_dataset': 
            xyz, rgb, labels = read_ply(data_file) 
            xyz = read_las(data_file)
            if voxel_width > 0: 
                xyz, rgb, labels = libply_c.prune(xyz, voxel_width, rgb, np.array(1,dtype='u1'), 0) 
                xyz = libply_c.prune(xyz, voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0] 
        graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof) 
        geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32') 
        del target_fea 
        sys.stdout.flush()
        print("    computing the superpoint graph...")

        if dataset=='s3dis':
            features = np.hstack((geof, (1.25*rgb/255.)-0.25)).astype('float32')#add rgb as a feature for partitioning N,4 + N,3 = N,7
            features[:,3] = 2. * features[:,3] 
            features = 0.5 * features
        elif dataset=='custom_dataset': 
                features = geof
                geof[:,3] = 2. * geof[:, 3]
            
        graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
        print("        minimal partition...") 
        components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                        , graph_nn["edge_weight"], reg_strength)

        return components,in_component

    
    # from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html and https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    def dbscan_partition(self,xyz,rgb,eps=0.5,dataset='s3dis',n_clusters=100):
        default_base = {'quantile': .3,
            'eps': .3,
            'damping': .9,
            'preference': -200,
            'n_neighbors': 10,
            'n_clusters': 3,
            'min_samples': 1,
            'xi': 0.05,
            'min_cluster_size': 0.1,
            'metric':"euclidean",
            'leaf_size':30,
            'n_init':10,
            'max_eps':np.inf,
            'threshold':0.5,
            'max_iter':100,
            'covariance_type':'full',
            'n_jobs':-1}
        def create_clusters(X,algo_params):
            params = default_base.copy()
            params.update(algo_params) 
            connectivity = kneighbors_graph(
                X, n_neighbors=params['n_neighbors'], include_self=False) 
            connectivity = 0.5 * (connectivity + connectivity.T) 
            self.ms = None
            self.two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            self.ward = cluster.AgglomerativeClustering(
                n_clusters=params['n_clusters'], linkage='ward',
                connectivity=connectivity)
            self.spectral = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors",n_jobs=params['n_jobs'])
            self.dbscan = cluster.DBSCAN(eps=params['eps'],min_samples=params['min_samples'],metric=params["metric"],leaf_size=params['leaf_size'],n_jobs=params['n_jobs'])
            self.optics = cluster.OPTICS(min_samples=params['min_samples'],
                                    xi=params['xi'],
                                    min_cluster_size=params['min_cluster_size'],n_jobs=params['n_jobs'])
            self.affinity_propagation = cluster.AffinityPropagation(
                damping=params['damping'], preference=params['preference'])
            self.average_linkage = cluster.AgglomerativeClustering(
                linkage="average", affinity="cityblock",
                n_clusters=params['n_clusters'], connectivity=connectivity)
            self.birch = cluster.Birch(n_clusters=params['n_clusters'],threshold=params['threshold'])
            self.gmm = mixture.GaussianMixture(
                n_components=params['n_clusters'], covariance_type=params['covariance_type']) 

        # euclidean with xyz and l2 with rgb
        def my_pairwise_distances(X, Y=None,n_jobs=-1): 
            return pairwise_distances(np.array([X[:3]]),np.array([Y[:3]]),metric="euclidean",n_jobs=n_jobs)+pairwise_distances(np.array([X[3:]]),np.array([Y[3:]]),metric="l2",n_jobs=n_jobs)

        if dataset=='s3dis': 
            xyz = xyz - xyz.mean(0).astype('f4')
            xyz[:,2] = xyz[:,2]*1.25
            ceiling_mask = xyz[:,2]>=0.9*max(xyz[:,2])
            xyz[ceiling_mask,2] = xyz[ceiling_mask,2]*5.0
            floor_mask = xyz[:,2]<=1.1*min(xyz[:,2])
            xyz[floor_mask,2] = xyz[floor_mask,2]/5.0
            xyz = xyz*1.25 
            rgb = (rgb).astype('f4')*1.25
            X = np.hstack((xyz,rgb))
        if not hasattr(self,'dbscan'):
            print('create clusters ...') 
            create_clusters(X,{'eps':eps,'n_clusters':n_clusters,'threshold':0.08,'random_state':0,'max_iter':10,'covariance_type':'full'})

        clustering = self.gmm.fit(X) 
        if hasattr(clustering, 'labels_'):
            return clustering.labels_
        else:
            return clustering.predict(X)

    
    def replace_label(self,inst_replace_method=1): 
        train_data_list, test_data_list = [], []
        xs = self.points_temp
        ys = self.labels_temp 
        print('################## making instance-level label noise ##################')
        if self.split == 'train.txt':
            avg_fake_per = 0
            if self.with_Cluster:
                instance_nums = []
            for i, y in enumerate(tqdm(ys)): # for each room 
                fake_per = 0.0 
                shape_y = ys[i].shape 
                try:
                    y = torch.from_numpy(y.reshape(-1,2))
                except:
                    y = y.reshape(-1,2) 
                if not self.copy_pre_cluster:
                    z = np.zeros(len(y))
                    n_bias = len(np.unique(y[:,1]))
                    for zid in range(len(z)):
                        z[zid] = (n_bias+1)*y[zid,0]+y[zid,1]
                        instance_nums.append(len(np.unique(z)))

                if inst_replace_method==1: 
                    if self.precent_NL==80: 
                        threshold = 0.87 # you may need to find different threshold for different random seed
                    elif self.precent_NL==60:
                        threshold = 0.852 # you may need to find different threshold for different random seed
                    elif self.precent_NL==40:
                        threshold = 0.8 # you may need to find different threshold for different random seed
                    while fake_per < threshold*(self.precent_NL/100.0): # 0.8 when 40 and 0.852 when 60 and 0.87 when 80
                        # rand find source class & bias
                        rand_class = np.random.randint(0, self.class_num)
                        rand_bias = np.random.randint(0, self.max_instances)
                        label = torch.from_numpy(np.array([rand_class,rand_bias]))
                        mask = label==y
                        instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                        while instance_mask.sum() == 0: # no sample with this class&bias in this room
                            rand_class = np.random.randint(0, self.class_num)
                            rand_bias = np.random.randint(0, self.max_instances)
                            label = torch.from_numpy(np.array([rand_class,rand_bias]))
                            mask = label==y
                            instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                        
                        # rand target class
                        target_class = np.random.randint(0, self.class_num)
                        while target_class == rand_class:
                            target_class = np.random.randint(0, self.class_num)
                        target_label = torch.from_numpy(np.array([target_class,0])) # bias is arbitrary
                        y[instance_mask] = target_label
                        fake_per += instance_mask.sum()/(y.shape[0]+0.0)
                elif inst_replace_method==2: # asymmetric
                    asymmetric_T = {5:2,2:5,11:6,6:11,10:7,7:10}# window \Leftrightarrow wall, board \Leftrightarrow door, sofa \Leftrightarrow chair
                    while fake_per < 0.67*(self.precent_NL/100.0): # you may need to find different threshold for different random seed
                        # rand find source class & bias
                        rand_class = np.random.randint(0, self.class_num)
                        rand_bias = np.random.randint(0, self.max_instances)
                        label = torch.from_numpy(np.array([rand_class,rand_bias]))
                        mask = label==y
                        instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                        while instance_mask.sum() == 0: # no sample with this class&bias in this room
                            rand_class = np.random.randint(0, self.class_num)
                            rand_bias = np.random.randint(0, self.max_instances)
                            label = torch.from_numpy(np.array([rand_class,rand_bias]))
                            mask = label==y
                            instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                
                        if rand_class in asymmetric_T.keys():# target class according to asymmetric_T
                            if np.random.randint(0, 3)<1: # with more than 33% according to asymmetric_T
                                target_class = asymmetric_T[rand_class]
                            else:
                                target_class = np.random.randint(0, self.class_num)
                        else:
                            target_class = np.random.randint(0, self.class_num)
                        while target_class == rand_class:
                            target_class = np.random.randint(0, self.class_num)
                        target_label = torch.from_numpy(np.array([target_class,0])) # bias is arbitrary
                        y[instance_mask] = target_label
                        fake_per += instance_mask.sum()/(y.shape[0]+0.0)
                
                elif inst_replace_method==3: # common asymmetric
                    asymmetric_T = {5:2,2:5,11:6,6:11,10:7,7:10}# door \Leftrightarrow wall, board \Leftrightarrow window, sofa \Leftrightarrow chair
                    cnt_total_points_in_asymmetric_T = 0
                    exist_classes_in_asymmetric_T,instcnt_for_exist_classes_in_asymmetric_T = [],[0,0,0,0,0,0,0,0,0,0,0,0] 
                    flip_cnt = self.max_instances
                    for ic in list(asymmetric_T.keys()):
                        mask = ic==y[:,0]
                        if sum(mask)>0:
                            exist_classes_in_asymmetric_T.append(ic)
                            instcnt_for_exist_classes_in_asymmetric_T[ic] += len(torch.unique(y[mask][:,1]))
                        cnt_total_points_in_asymmetric_T += sum(mask)
                    if len(exist_classes_in_asymmetric_T) == 0:
                        continue # skip this room
                    while fake_per < 0.84*(self.precent_NL/100.0): # you may need to find different threshold for different random seed
                        print(fake_per,exist_classes_in_asymmetric_T,instcnt_for_exist_classes_in_asymmetric_T)
                        # rand find source class & bias 
                        rand_class = random.choice(exist_classes_in_asymmetric_T) 
                        rand_bias = random.choice(list(torch.unique(y[rand_class==y[:,0]][:,1]).numpy()))
                        label = torch.from_numpy(np.array([rand_class,rand_bias]))
                        mask = label==y
                        instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                        while instance_mask.sum() == 0: # no sample with this class&bias in this room
                            rand_class = random.choice(exist_classes_in_asymmetric_T)
                            rand_bias = random.choice(list(torch.unique(y[rand_class==y[:,0]][:,1]).numpy()))
                            label = torch.from_numpy(np.array([rand_class,rand_bias]))
                            mask = label==y
                            instance_mask = torch.logical_and(mask[:,0],mask[:,1])
                        target_class = asymmetric_T[rand_class]
                        # note that in common asymmetric, fake_per only counts classes in asymmetric_T
                        target_label = torch.from_numpy(np.array([target_class,flip_cnt])) # bias is flip_cnt
                        flip_cnt += 1
                        y[instance_mask] = target_label
                        fake_per += instance_mask.sum()/(cnt_total_points_in_asymmetric_T+0.0)
                        
                        instcnt_for_exist_classes_in_asymmetric_T[rand_class] -= 1
                        if instcnt_for_exist_classes_in_asymmetric_T[rand_class]<1:
                            for i,x in enumerate(exist_classes_in_asymmetric_T):
                                if x == rand_class:
                                    del exist_classes_in_asymmetric_T[i] 
                y=y.reshape(shape_y)
                ys[i] = y.clone()
                avg_fake_per += fake_per/(len(ys)+0.0)
            print('!!!!!!!!!!!!!!!!!! avg inst fake_per !!!!!!!!!!!!!!!!!: ',avg_fake_per)
                            




        if self.split == 'train.txt' and self.copy_pre_cluster: 
            from torch_geometric.datasets import S3DIS
            data_with_cluster = S3DIS(root=self.out_root,train=self.split == 'train.txt')
            sample_cnt = 0


        for i, (x, y) in enumerate(tqdm(zip(xs, ys))):
            if self.split == 'train.txt' and self.copy_pre_cluster: 
                z_cluster_list = []
                for sampleid in range(sample_cnt,(sample_cnt+ys[i].shape[0])): 
                    z_cluster_list.append(data_with_cluster[sampleid].z) 
                assert torch.all(data_with_cluster[sample_cnt].pos[0] == torch.tensor(xs[i][0,:,:3][0]))
                sample_cnt += ys[i].shape[0] 

                y_shape = y.shape
                y = y.reshape(-1,2)
                y[:,1] = torch.from_numpy(np.hstack(z_cluster_list).astype('float64')).long()
                y = y.reshape(y_shape)

            elif self.split == 'train.txt' and self.with_Cluster and self.ClusterByPartitionMethods:
                x_each_room,y_each_room = x.reshape(-1,9),y.reshape(-1,2) 
                components,in_component = self.partition(xyz=x_each_room[:, 6:],  rgb=x_each_room[:, 3:6], labels=y_each_room[:,0],reg_strength = self.reg_strength) 
                print(len(components))
                y_shape = y.shape
                y = y.reshape(-1,2)
                y[:,1] = torch.from_numpy(in_component.astype('float64')).long()
                y = y.reshape(y_shape) 
            elif self.split == 'train.txt' and self.with_Cluster and self.ClusterByDBSCAN:
                x_each_room,y_each_room = x.reshape(-1,9),y.reshape(-1,2) 
                max_x_length = 100 # limited by 30G ram
                # max_x_length = 25 # limited by 10G ram
                if len(x) < max_x_length:
                    in_component = self.dbscan_partition(xyz=x_each_room[:, 6:],  rgb=x_each_room[:, 3:6], eps = self.MaxNeighborDistance,n_clusters=3*instance_nums[i] if self.ClusterByGMM else 100)
                else:
                    in_component_list,in_component_max = [],0
                    for ith in range(len(x)//max_x_length):
                        print(ith,'-th in ',len(x))
                        x_each_room,y_each_room = x[max_x_length*ith:max_x_length*(ith+1),:,:].reshape(-1,9),y[max_x_length*ith:max_x_length*(ith+1),:,:].reshape(-1,2) 
                        in_component = self.dbscan_partition(xyz=x_each_room[:, 6:],  rgb=x_each_room[:, 3:6], eps = self.MaxNeighborDistance,n_clusters=3*instance_nums[i] if self.ClusterByGMM else 100)
                        in_component_list.append(in_component+in_component_max)
                        in_component_max += max(in_component)
                    if (len(x)%max_x_length)!=0:  
                        x_each_room,y_each_room = x[max_x_length*(len(x)//max_x_length):,:,:].reshape(-1,9),y[max_x_length*(len(x)//max_x_length):,:,:].reshape(-1,2) 
                        in_component = self.dbscan_partition(xyz=x_each_room[:, 6:],  rgb=x_each_room[:, 3:6], eps = self.MaxNeighborDistance,n_clusters=3*instance_nums[i] if self.ClusterByGMM else 100)
                        in_component_list.append(in_component+in_component_max)
                    in_component = np.hstack(in_component_list)
                print(in_component,max(in_component),min(in_component),' total #points:',len(in_component),' #-1s:',sum(in_component==-1))
                y_shape = y.shape
                y = y.reshape(-1,2)
                y[:,1] = torch.from_numpy(in_component.astype('float64')).long()
                y = y.reshape(y_shape)



            for x_4096,y_4096 in zip(torch.from_numpy(x), y): 
                try:
                    y_4096 = torch.from_numpy(y_4096)
                except:
                    pass
                if self.with_Cluster: 
                    if self.split == 'train.txt':
                        data = Data(pos=x_4096[:, :3], x=x_4096[:, 3:], y=y_4096[:,0].long(), z=y_4096[:,1].long())
                    else:
                        data = Data(pos=x_4096[:, :3], x=x_4096[:, 3:], y=torch.from_numpy(y_4096)[:,0].long(), z=torch.from_numpy(y_4096)[:,1].long())
                else:
                    if self.split == 'train.txt':
                        data = Data(pos=x_4096[:, :3], x=x_4096[:, 3:], y=y_4096[:,0].long())
                    else:
                        data = Data(pos=x_4096[:, :3], x=x_4096[:, 3:], y=torch.from_numpy(y_4096)[:,0].long())

                if self.split == 'train.txt':
                    train_data_list.append(data)
                if self.split == 'test.txt':
                    test_data_list.append(data)
                new_file_name = self.rooms[i][:-3]
                file1 = open(self.log_file,"a+") 
                file1.writelines(new_file_name+'\n') 
                file1.close()

        del self.points_temp,self.labels_temp,self.coords_temp,xs,ys,x,y,x_4096,y_4096
        return train_data_list,test_data_list 
 




class save_S3DIS(InMemoryDataset):
    def __init__(self):
        pass
    def save_noisy_label(self,train_data_list,test_data_list,training,out_processed_paths):
        print('save processed torch .pt file ...')
        if training:
            torch.save(self.collate(train_data_list), out_processed_paths[0])
        else:
            torch.save(self.collate(test_data_list), out_processed_paths[1])

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="store_true",
        default=False, 
    ) 
    parser.add_argument(
        "--root",
        default='data_with_ins_label', 
        help='path for clean labeled data',
    ) 
    parser.add_argument(
        "--replace_method", 
        default=1, 
        help='Symmetry=1, Asymmetry=2, common Asymmetry=3',
    ) 
    parser.add_argument(
        "--precent_NL", 
        default=80, 
    )
    parser.add_argument(
        "--pre_cluster_path", 
        default=None, 
        help='given a pre-clustered dataset dir, it will copy the clusters to the new generated data without doing clustering again',
    ) 

    args = parser.parse_args()
    return args



args = parse_args()
training=args.training
inst_replace_method = args.replace_method
pre_cluster_path = args.pre_cluster_path
precent_NL = args.precent_NL
root = args.root
S3DIS_ins = S3DIS_instance(root=root,training=training,pre_cluster_path=pre_cluster_path,precent_NL=precent_NL) 
print('The max instances num is: ',S3DIS_ins.max_instances) 
train_data_list,test_data_list = S3DIS_ins.replace_label(inst_replace_method=1)
out_processed_paths = S3DIS_ins.out_processed_paths
del S3DIS_ins
save_S3DIS_ins = save_S3DIS()
save_S3DIS_ins.save_noisy_label(train_data_list,test_data_list,training=training,out_processed_paths=out_processed_paths)


