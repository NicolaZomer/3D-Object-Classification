"""
This implements a dataset for voxelized point clouds

Nevertheless, using off files is computationaly expensive, therefore we use
the approach proposed in https://github.com/MonteYang/VoxNet.pytorch
to convert files into binvox
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
import pandas as pd
import open3d as o3d

class VoxelDataset(Dataset):
    def __init__(self, 
                dataset_folder, 
                transform=None, 
                train=True, 
                file_extension='.off', 
                ndata=100,
                shape = (32,32,32),
                voxel_size = 1
                ):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        modelnet is organized as follows:
        the folder dataset_folder contains a folder for each class
        each class folder contains two more folders: train and test
        each train/test folder contains a all the point clouds for that class
        in summary, the folder structure is:

        dataset_folder
          |__ class1  
          |   |__ train   
          |   |   |__ cloud1.off
          |   |   |__ cloud2.off
          |   |   |__ ...
          |   |__ test
          |       |__ cloud1.off
          |       |__ cloud2.off
          |       |__ ...
          |__ class2
          |   |__ ...
          |__ ...

        """
        # extension
        self.file_extension = file_extension

        # all classes 
        classes = os.listdir(dataset_folder)
        for c in classes:
            if not os.path.isdir(os.path.join(dataset_folder, c)):
                classes.remove(c)
        classes.sort()
        self.classes = classes

        # all point clouds
        self.paths = []
        self.label_names = []
        self.labels = []

        if train:
            for i, c in enumerate(self.classes):
                all_files = glob.glob(os.path.join(dataset_folder, c, 'train', "*"+file_extension))

                self.paths += all_files
                self.label_names += [c] * len(all_files)
                self.labels += [i] * len(all_files)
        else:
            for i, c in enumerate(self.classes):
                all_files = glob.glob(os.path.join(dataset_folder, c, 'test', "*"+file_extension))
                self.paths += all_files
                self.label_names += [c] * len(all_files)
                self.labels += [i] * len(all_files)

        # shuffle
        idx = np.arange(len(self.paths))
        np.random.shuffle(idx)
        self.paths = np.array(self.paths)[idx]
        self.label_names = np.array(self.label_names)[idx]
        self.labels = np.array(self.labels)[idx]

        # subsample
        if ndata is not None:
            self.paths = self.paths[:ndata]
            self.label_names = self.label_names[:ndata]
            self.labels = self.labels[:ndata]

        # print (self.paths)
        self.transform = transform

        self.shape = shape
        self.voxel_size = voxel_size
        print (f'Loaded {len(self.paths)} point clouds from {len(self.classes)} classes')
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        # parsing is specific to the file extension
        x = o3d.io.read_triangle_mesh(path)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(x, voxel_size=self.voxel_size)
        
        # crop the voxel grid to the bounding box (32,32,50)
        #fill from the center with radial distance
        voxels = voxel_grid.get_voxels()
        voxels = [v.grid_index for v in voxels]
        voxels = np.array(voxels)

        # create the  matrix with the voxel grid
        bounding_box = voxel_grid.get_axis_aligned_bounding_box()
        x, y, z = bounding_box.get_max_bound() - bounding_box.get_min_bound()
        grid = np.zeros((int(x/self.voxel_size), int(y/self.voxel_size), int(z/self.voxel_size)))

        # fill the grid
        for v in voxels:
            grid[v[0], v[1], v[2]] = 1

        # crop the grid to the input size, centered
        input_size = (32,32,32)
        x, y, z = grid.shape
        x0 = int((x - input_size[0])/2)
        y0 = int((y - input_size[1])/2)
        z0 = int((z - input_size[2])/2)

        voxel = grid[x0:x0+input_size[0], y0:y0+input_size[1], z0:z0+input_size[2]]
        sample = (voxel, label)
    
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':

    voxel_size = 1
    path = 'dataset/ModelNet40/flower_pot/test/flower_pot_0152.off'
    path = 'modelnet40_normal_resampled/bench/test/bench_0008.txt'
    x = o3d.io.read_triangle_mesh(path)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(x, voxel_size=voxel_size)

    # create the  matrix with the voxel grid
    bounding_box = voxel_grid.get_axis_aligned_bounding_box()
    x, y, z = bounding_box.get_max_bound() - bounding_box.get_min_bound()
    grid = np.zeros((int(x/voxel_size), int(y/voxel_size), int(z/voxel_size)))
    # crop the voxel grid to the bounding box (32,32,50)
    #fill from the center with radial distance
    voxels = voxel_grid.get_voxels()
    voxels = [v.grid_index for v in voxels]
    voxels = np.array(voxels)

    # fill the grid
    for v in voxels:
        grid[v[0], v[1], v[2]] = 1

    # crop the grid to the input size, centered
    input_size = (32,32,32)
    x, y, z = grid.shape
    x0 = int((x - input_size[0])/2)
    y0 = int((y - input_size[1])/2)
    z0 = int((z - input_size[2])/2)

    voxel = grid[x0:x0+input_size[0], y0:y0+input_size[1], z0:z0+input_size[2]]