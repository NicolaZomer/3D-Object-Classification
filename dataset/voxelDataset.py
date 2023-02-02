# coding: utf-8
import os
import sys
import numpy as np
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('..')
from mapping import index2class, class2index
from binvox_utils.binvox_rw import read_as_3d_array

class VoxelDataset(Dataset):
    def __init__(self, dataset_folder, train=True):
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
        self.dataset_folder = dataset_folder
        self.samples_str = []
        self.cls2idx = {}
        if train: split = 'train'
        else: split = 'test'

        # all classes 
        classes = os.listdir(dataset_folder)
        for c in classes:
            if not os.path.isdir(os.path.join(dataset_folder, c)):
                classes.remove(c)
        classes.sort()
        self.classes = classes 
        self.n_classes = len(classes)

        self.cls2idx = class2index#{cls: idx for idx, cls in enumerate(classes)}
        for v in self.classes:
            for sample_str in glob.glob(os.path.join(dataset_folder, v, split, '*.binvox')):
                if re.match(r"[a-zA-Z]+_\d+.binvox", os.path.basename(sample_str)):
                    self.samples_str.append(sample_str)


    def __getitem__(self, idx):
        sample_name = self.samples_str[idx]
        cls_name = re.split(r"_\d+\.binvox", os.path.basename(sample_name))[0]
        cls_idx = self.cls2idx[cls_name]
        with open(sample_name, 'rb') as file:
            data = np.int32(read_as_3d_array(file).data)
            data = data[np.newaxis, :]

        # torch tensor int
        data = torch.from_numpy(data).type(torch.float32)
        
        sample = (data, cls_idx)
        return sample

    def __len__(self):
        return len(self.samples_str)


if __name__ == "__main__":
    
    dataset_folder = 'ModelNet40'
    dataset = VoxelDataset(dataset_folder, Train=True)
    cnt = len(dataset)

    data, cls_idx = dataset[0][0], dataset[0][1]
    print ('number of voxels: ', np.sum(data))

    # print(f"length: {cnt}\nsample data: {data}\nsample cls: {cls_idx}")