"""
plot clusters obtained with tsne to show that codewords allow to separate the data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
sns.set_theme(palette='Dark2', style='whitegrid', font_scale=1.5)

from sklearn.manifold import TSNE


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='foldingnet')
args = parser.parse_args()

def main (
    model_name='foldingnet', 
    ):

    if model_name == 'foldingnet':
        data_dir = '../dataset/svm_dataset_fold'
    elif model_name == 'vox':
        data_dir = '../dataset/svm_dataset_vox'

    data_train = np.load(os.path.join(data_dir, 'encoded_states_train.npy'))
    labels_train = np.load(os.path.join(data_dir, 'labels_train.npy'))

    data_test = np.load(os.path.join(data_dir, 'encoded_states_val.npy'))
    labels_test = np.load(os.path.join(data_dir, 'labels_val.npy'))

    # get a subset of 10 classes
    classes = range (0, 40, 4)
    mask = np.isin(labels_train, classes)

    data_train = data_train[mask]
    labels_train = labels_train[mask]

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(data_train)

    # create cmap with 40 colors
    from matplotlib.colors import ListedColormap
    ncolors = 40

    # add 10 colors from each palette
    cmap = ListedColormap(sns.color_palette("hls", ncolors))


    print (tsne_results.shape)
    plt.scatter (tsne_results[:, 0], tsne_results[:, 1], c=labels_train, cmap=cmap)
    plt.title(f'TSNE on {len(classes)} classes')
    plt.show()

if __name__ == '__main__':
    main(model_name=args.model_name)