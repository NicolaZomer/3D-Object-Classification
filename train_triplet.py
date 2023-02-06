import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import  recall_score, precision_score, f1_score, accuracy_score

from dataset.PointCloudDataset import PointCloudDataset
from dataset.voxelDataset import VoxelDataset
from classifier import Classifier

#from networks.PointNet import PointNet
#from networks.voxnet import VoxNet
#from networks.res_voxnet import ResVoxNet

from TripletLoss.tripletnet import TripletNet
from TripletLoss.custom_losses import TripletCenterLoss


import argparse
import os
# 2000 dati allenati con batch 32 e 5e-4
# 4000 dati allenati con batch 64 e 1e-3

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--model_name', type=str, default='tripletnet', help='model name (default: tripletnet)', choices=['tripletnet'])
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 32)')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints (default: checkpoints/pointnet)')
parser.add_argument('--ndata', type=int, default=4000, help='number of data points to use (default: 2000)')
parser.add_argument('--npoints', type=int, default=5000, help='number of points in the point cloud (default: 4000)')
parser.add_argument('--train', type=bool, default=True, help='train or test (default: True)')
parser.add_argument('--rotation', type=bool, default=True, help='augment samples using random rotations (default: False)')
args = parser.parse_args()


def main (
    model_name='tripletnet',
    epochs=10,
    lr=0.001,
    batch_size=32,
    save_dir='checkpoints',
    ndata=4000,
    npoints = 4000,
    train=False,
    rotation=False
    ):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    save_dir = os.path.join(save_dir, model_name)

   


    ##############################
    # LOAD MODEL AND DATASET
    ##############################
    print (f"Loading model {model_name}...")
    if model_name == 'tripletnet':

        dataset_train  = PointCloudDataset('dataset/modelnet40_normal_resampled', 
                                            train=True, 
                                            ndata=ndata,
                                            file_extension='.txt',
                                            npoints=npoints,
                                            rotation=rotation
                                        )
        
        if train: test_data = int(ndata/20)
        else: test_data = -1
        dataset_val    = PointCloudDataset('dataset/modelnet40_normal_resampled', 
                                            train=False, 
                                            ndata=test_data,
                                            file_extension='.txt', 
                                            npoints=npoints,
                                            rotation=False
                                        )

        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")
    
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

        Net = TripletNet(nclasses=40)
        parameters = Net.parameters()
        optimizer =  optim.SGD(parameters, lr=5e-4, weight_decay=1e-5)


    else:
        raise ValueError(f"Model {model_name} not implemented")


    # load model if exists
    models_saved = glob.glob(os.path.join(save_dir, 'model_*.torch'))
    if len(models_saved) > 0:
        # get most recent model
        epoches_done = max([int(model.split('_')[-1].split('.')[0]) for model in models_saved])
        model_path = os.path.join(save_dir, f'model_{epoches_done}.torch')
        print(f"Loading model from {model_path}")
        Net.load_state_dict(torch.load(model_path))
    else:
        epoches_done = 0


    # move model to device
    Net.to(device)
    classifier = Classifier(Net, device=device)
    
    #LOSS
    loss_cls = nn.CrossEntropyLoss()
    loss_tcl = TripletCenterLoss(margin=5)

    if train:
        print ("Starting training")
        classifier.train_triplet(dataloader_train, dataloader_val, epochs=epochs, optimizer=optimizer, loss_fn = [loss_cls, loss_tcl],
                         save_dir=save_dir, start_epoch=epoches_done+1)
        print ("Training done")
    else:
        print ("Starting testing")
        results = classifier.test(dataloader_val)
        print ("Testing done")
        print(results)

        # save results in csv
        results.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)




if __name__ == '__main__':
    main(
       **vars(args)
    )