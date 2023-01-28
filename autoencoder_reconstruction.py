import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    compression_ratios = []
    for ref_cloud in tqdm(train_loader):
        ref_cloud = ref_cloud.to(DEVICE)

        optimizer.zero_grad()
        encoded, decoded = model(ref_cloud.permute(0,2,1).contiguous())

        loss = loss_fn(ref_cloud.permute(0,2,1).contiguous(), decoded)

        #product shapes normalized over batch_size
        encoding_size = encoded.shape[1]
        ref_cloud_size = ref_cloud.shape[1] * ref_cloud.shape[2]
        
        compression_ratio = encoding_size / ref_cloud_size
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        compression_ratios.append(compression_ratio)
    
    results = {
        'loss': np.mean(losses),
        'compression_ratio': np.mean(compression_ratios)
    }
    return results


def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []
    compression_ratios = []

    with torch.no_grad():
        for ref_cloud in tqdm(test_loader):
            ref_cloud = ref_cloud.to(DEVICE)
            encoded, decoded = model(ref_cloud.permute(0,2,1).contiguous())

            loss = loss_fn(ref_cloud.permute(0,2,1).contiguous(), decoded)

            #product shapes normalized over batch_size
            encoding_size = encoded.shape[1]
            ref_cloud_size = ref_cloud.shape[1] * ref_cloud.shape[2]
            
            compression_ratio = encoding_size / ref_cloud_size

            losses.append(loss.item())
            compression_ratios.append(compression_ratio) 
    model.train()
    results = {
        'loss': np.mean(losses),
        'compression_ratio': np.mean(compression_ratios)
    }
    return results