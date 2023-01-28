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
class PointCloudDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, train=True, file_extension='.off'):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        # modelnet is organized as follows:
        # the folder dataset_folder contains a folder for each class
        # each class folder contains two more folders: train and test
        # each train/test folder contains a all the point clouds for that class
        # in summary, the folder structure is:

        # dataset_folder
        #   |__ class1  
        #   |   |__ train   
        #   |   |   |__ cloud1.off
        #   |   |   |__ cloud2.off
        #   |   |   |__ ...
        #   |   |__ test
        #   |       |__ cloud1.off
        #   |       |__ cloud2.off
        #   |       |__ ...
        #   |__ class2
        #   |   |__ ...
        #   |__ ...

        # all classes 
        self.classes = os.listdir(dataset_folder)
        self.classes.sort()

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
        print (f'Loaded {len(self.paths)} point clouds from {len(self.classes)} classes')

        # print (self.paths)
        self.transform = transform
        
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.clouds[idx]
        label = self.labels[idx]
        x = o3d.io.read_triangle_mesh(path)
        x = np.asarray(x.vertices)
        # rehsape to num_pointsx3
        x = np.reshape(x, (-1, 3))
        # torch
        x = torch.from_numpy(x).float()

        sample = (x, label)
        if self.transform:
            sample = self.transform(sample)
        return sample