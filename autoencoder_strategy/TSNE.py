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
        data_dir = '../dataset/svm_dataset_foldingnet'
    elif model_name == 'vox':
        data_dir = '../dataset/svm_dataset_vox'

    data_train = np.load(os.path.join(data_dir, 'encoded_states_train.npy'))
    labels_train = np.load(os.path.join(data_dir, 'labels_train.npy'))

    data_test = np.load(os.path.join(data_dir, 'encoded_states_val.npy'))
    labels_test = np.load(os.path.join(data_dir, 'labels_val.npy'))

    # get a subset of 10 classes
    sys.path.append('../')
    from mapping import class2index, index2class

    classes = { 'airplane',   'bowl', 'chair', 'desk', 'table', 'guitar', 'car',
                'flower_pot', 'laptop', 
                 'piano',  }

    classes = [class2index[c] for c in classes]
    mask = np.isin(labels_train, classes)

    data_train = data_train[mask]
    labels_train = labels_train[mask]
    label_classes = [index2class[l] for l in labels_train]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
    tsne_results = tsne.fit_transform(data_train)
    import pandas as pd
    results = pd.DataFrame( {'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'label': label_classes} )

    # create cmap with 40 colors
    from matplotlib.colors import ListedColormap

    # add 10 colors from each palette
    import matplotlib as mpl
    colors = [
        'red',  'magenta','green', 'orange', 'gold', 'blue',  'moccasin',
        'cornflowerblue',   'blueviolet', 'firebrick',
        'orchid', 'navy', 'maroon', 'moccasin', 'indigo', 'hotpink', 'gold',  'darkslategray', 
    ]
    ncolors = len(classes)
    cmap = mpl.colors.ListedColormap(colors[:ncolors])
    cmap = sns.set_palette(sns.color_palette(colors))

    print (tsne_results.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter (tsne_results[:, 0], tsne_results[:, 1], c=labels_train, cmap=cmap,
    sns.scatterplot(data=results, x='x', y='y', hue='label',ax=ax, s=80, palette=cmap,  linewidth=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax.set_title(f'TSNE')
    ax.set_xlabel('encoded var. 1')
    ax.set_ylabel('encoded var. 2')
    #save
    fig.savefig(f'../imgs/tsne_{model_name}.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main(model_name=args.model_name)