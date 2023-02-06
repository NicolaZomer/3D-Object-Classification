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

def rotate_point_cloud(xyz):
    angle = np.random.choice( np.arange(-np.pi/2, np.pi/2 +  np.pi/4, np.pi/4) )       
    R = xyz.get_rotation_matrix_from_xyz((0, 0, angle))

    xyz = xyz.rotate(R, center=(0,0,0))
    return xyz


class PointCloudDataset(Dataset):
    def __init__(self, 
                dataset_folder, 
                transform=None, 
                train=True, 
                file_extension='.off', 
                ndata=100,
                npoints=10000,
                rotation=False,
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
        
        self.rotation = rotation
        


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

        self.npoints = npoints
        print (f'Loaded {len(self.paths)} point clouds from {len(self.classes)} classes')
    
    def __len__(self):
        return len(self.paths)
    
    

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        # parsing is specific to the file extension
        if self.file_extension == '.off':
            x = o3d.io.read_triangle_mesh(path)
            if self.rotation:
                x = rotate_point_cloud(x)
                
            x = np.asarray(x.vertices)
            # rehsape to num_pointsx3
            x = np.reshape(x, (-1, 3))
        
        elif self.file_extension == '.txt':
            x = np.loadtxt(path, delimiter=',')
            # keep first 3 columns
            x = x[:, :3]
            # rehsape to num_pointsx3
            x = np.reshape(x, (-1, 3))

        # get npoints shuffle
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        x = x[idx[:self.npoints], :]
        
        
        
        # torch
        x = torch.from_numpy(x).float()
        sample = (x, label)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    