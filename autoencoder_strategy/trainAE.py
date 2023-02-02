"""
The purpose of this strategy is to train an AE on the reconstruction task, in order to learn
a feature representation of the input data. 
The encoded features are then used as input to a classifier.

We first test the pointcloud dataset: folding net as AE for point cloud reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import numpy as np
import matplotlib.pyplot as plt
import os, glob
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(train_loader, model, optimizer, loss_fn):
    losses = []
    for ref_cloud in tqdm(train_loader):
        ref_cloud = ref_cloud[0] # the second element in the list is the label, but we dont need it now 
        ref_cloud = ref_cloud.to(DEVICE)
        optimizer.zero_grad()
        decoded, encoded = model(ref_cloud)
        # get_loss is a function of net
        loss = loss_fn(ref_cloud, decoded)
        # loss = model.get_loss(ref_cloud, decoded)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        for ref_cloud in tqdm(test_loader):
            ref_cloud = ref_cloud[0]
            ref_cloud = ref_cloud.to(DEVICE)

            decoded, encoded = model(ref_cloud)
            loss = loss_fn(ref_cloud, decoded)
            # loss = model.get_loss(ref_cloud, decoded)
            losses.append(loss.item())
    model.train()
    
    return np.mean(losses)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--checkpoints_path', type=str, default='../checkpoints')
parser.add_argument('--ndata', type=int, default=1000, help='Number of data ')
parser.add_argument('--npoints', type=int, default=4000, help='Number of points per cloud')
parser.add_argument('--train', type=bool, default=True, help='Train or test')
parser.add_argument('--model_name', type=str , default='foldingnet', help='foldingnet or vox', choices=['foldingnet', 'vox'])

args = parser.parse_args()


def main(
    batch_size=32,
    epochs=100,
    checkpoints_path='../checkpoints',
    ndata=4000,
    npoints=5000,
    train=True,
    model_name='foldingnet'
    ):
    checkpoints_path = os.path.join(checkpoints_path, model_name)

    ############## LOAD MODEL AND DEFINE LEARNING SCENARIO ################
    from FoldingNet import FoldNet, Encoder, Decoder
    from voxel_ae import voxAutoEncoder
    from ch_loss import ChamferLoss


    import pandas as pd
    test_losses, train_losses = [], []

    print ('='*20, 'LOADING MODEL', '='*20)
    print (f"Model path {checkpoints_path}")
    ############## LOAD PREVIOUS INFORMATIONS ################
    # load results if exists
    if os.path.exists(os.path.join(checkpoints_path, 'train_loss.txt')):
        train_losses = np.loadtxt(os.path.join(checkpoints_path, 'train_loss.txt'))
        train_losses = train_losses.tolist()

    if os.path.exists(os.path.join(checkpoints_path, 'val_loss.txt')):
        test_losses = np.loadtxt(os.path.join(checkpoints_path, 'val_loss.txt'))
        test_losses = test_losses.tolist()

    # create save folder if not exists
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)


    ############## LOAD DATA ################
    import sys
    sys.path.append('../')
    from dataset.PointCloudDataset import PointCloudDataset
    from dataset.voxelDataset import VoxelDataset


    print ('='*20, 'LOADING DATA', '='*20)

    if model_name == 'foldingnet':
        dataset_train  = PointCloudDataset('../dataset/modelnet40_normal_resampled', 
                                                train=True, 
                                                ndata=4000,
                                                file_extension='.txt', 
                                                npoints=npoints
                                            )
        if train: test_data = int(ndata/20)
        else: test_data = -1
        dataset_val    = PointCloudDataset('../dataset/modelnet40_normal_resampled', 
                                            train=False, 
                                            ndata=2000,
                                            file_extension='.txt', 
                                            npoints=npoints
                                        )

        # sample ndata points from each cloud
        np.random.seed(0)
        dataset_train = torch.utils.data.Subset(dataset_train, np.random.choice(len(dataset_train), ndata, replace=False))
        dataset_val = torch.utils.data.Subset(dataset_val, np.random.choice(len(dataset_val), ndata, replace=False))    

        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")

        model = FoldNet(num_points=npoints).to(DEVICE)
    
    elif model_name == 'vox':
        input_shape = (32, 32, 32)
        dataset_train  = VoxelDataset('../dataset/ModelNet40', 
                                        train=True, 
                                    )
        dataset_val    = VoxelDataset('../dataset/ModelNet40', 
                                        train=False, 
                                        )


        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")
        
        model = voxAutoEncoder(input_shape).to(DEVICE)


    # load model if exists
    models_saved = glob.glob(os.path.join(checkpoints_path, 'model_*.pth'))
    if len(models_saved) > 0:
        # get most recent model
        epoches_done = max([int(m.split('_')[-1].split('.')[0]) for m in models_saved])
        model_path = os.path.join(checkpoints_path, f'model_{epoches_done}.pth')
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        epoches_done = 0

   
    from torch.utils.data import DataLoader, SubsetRandomSampler
    ndata = 1000
    # subset of dataloader
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, sampler=SubsetRandomSampler(range(ndata)))
    test_loader = DataLoader(dataset_val, batch_size=32,  sampler=SubsetRandomSampler(range(ndata)))

    ############## TRAINING ################
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_fn = None
    if model_name == 'foldingnet':
        print ('='*20, 'USING CHAMFER LOSS', '='*20)
        loss_fn = ChamferLoss()
    
    elif model_name == 'vox':
        print ('='*20, 'USING MSE LOSS', '='*20)
        loss_fn = nn.MSELoss()

    if train:
        print ('='*20, 'TRAINING', '='*20)
        for epoch in range(epoches_done, epoches_done+epochs+1):
            print('=' * 20, epoch + 1, '=' * 20)
            tloss = train_one_epoch(train_loader, model, optimizer, loss_fn)
            vloss= test_one_epoch(test_loader, model, loss_fn)
            print('Epoch: {}, train loss: {:.4f}, val loss: {:.4f}'.format(epoch, tloss, vloss))

            train_losses.append(tloss)
            test_losses.append(vloss)
            
            # save model
            saved_path = os.path.join(checkpoints_path, "model_{}.pth".format(epoch))
            torch.save(model.state_dict(), saved_path)

            # save losses 
            np.savetxt(os.path.join(checkpoints_path, 'train_loss.txt'), train_losses)
            np.savetxt(os.path.join(checkpoints_path, 'val_loss.txt'), test_losses)

    else:
        print ('=' * 20, 'TESTING', '=' * 20)
        loss = test_one_epoch(test_loader, model)
        print('Test loss: {:.4f}'.format(loss))



if __name__ == '__main__':
    main( **vars(args))