import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
sys.path.append('../')
from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform

mapping = { #old_new
    1: 0,
    7: 1,
    8: 2,
    9: 3,
    10: 4,
    100: 5,
}


class SELMA(Dataset):
    def __init__(self, root, npts=4000, train=True,  nfiles=None, data_per_class=200):
        super(SELMA, self).__init__()

        # find all  directories in root
        dirname = 'train_data' if train else 'val_data'
        all_dirs = [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        paths = [os.path.join(root, item, dirname) for item in all_dirs]
        # print ('paths', paths)
        self.classes = []

    
        self.train = train

        self.files = []
        for path in paths:
            label = int(path.split('/')[-2].split('_')[-1])
            self.files += [os.path.join(path, item) for item in sorted(os.listdir(path))][:data_per_class]
            self.classes += [label]*data_per_class
            if nfiles is not None:
                self.files = self.files[:nfiles]
                
            # only if point cloud is not empty
            self.files = [item for item in self.files if readpcd(item).has_points()]
            self.npts = npts
        
        # remap labels
        self.classes = [mapping[item] for item in self.classes]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        ref_cloud = readpcd(file, rtype='npy')
        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        ref_cloud = pc_normalize(ref_cloud)
        ref_cloud = torch.from_numpy(ref_cloud).float()

        label = self.classes[item]
        label = torch.from_numpy(np.array(label)).long()
        sample = (ref_cloud, label)
       
        return sample
        
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    root = '../dataset_autoencoder_labels'
    dataset = SELMA(root, train=True)
    
    print (len(dataset))
    print (dataset[0][0].shape)
    print (dataset[0][1])
    