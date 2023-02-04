import argparse
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.selma import SELMA
from utils import time_calc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# soppress user warnings
import warnings
warnings.filterwarnings("ignore")
def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root',help='the data path', default='dataset_final')
    parser.add_argument('--load', type=bool, default=True,help='whether to load the trained model')
    parser.add_argument('--train_npts', type=int,  default=4000, help='the points number of each pc for training')
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gn', action='store_true', help='whether to use group normalization')
    parser.add_argument('--epoches', type=int, default=40)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_dim', type=int, default=3, help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8, help='iteration nums in one model forward')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=[50, 250], help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='models',help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=1,help='the frequency to save the logs and checkpoints')
    args = parser.parse_args()
    return args


import sys

@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    for sample in tqdm(train_loader):
        ref_cloud = sample[0].to(device)
        label = sample[1].to(device)

        optimizer.zero_grad()
        out = model(ref_cloud)
        loss = loss_fn(out, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()    

    return np.mean(losses)


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []

    with torch.no_grad():
        for sample in tqdm(test_loader):
            ref_cloud = sample[0].to(device)
            label = sample[1].to(device)

            out = model(ref_cloud)
            loss = loss_fn(out, label)
            losses.append(loss.item())

    return np.mean(losses)


def main():
    args = config_params()
    print(args)

    print ('Setting up data...')
    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    train_set = SELMA('dataset_autoencoder_labels', args.train_npts)
    test_set = SELMA('dataset_autoencoder_labels', args.train_npts, False)
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    # test dataset
    print ('test dataset', len(test_set))
    print ('train dataset', len(train_set))

    # load model if exists from checkpoints
    import glob
    model_paths = glob.glob(os.path.join(checkpoints_path, 'model_*.pth'))
    model_paths.sort()
    epochs = [int(path.split('_')[-1].split('.')[0]) for path in model_paths]
    start_epoch = max(epochs) if len(epochs) > 0 else 0


    sys.path.append('networks')
    from networks.PointNet import PointNet

    model = PointNet(nclasses=6)
    parameters = model.parameters()
    optimizer =  torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)


    if len(model_paths) > 0 and args.load:
        model_path = os.path.join(checkpoints_path, 'model_{}.pth'.format(start_epoch))
        print('load model from {}'.format(model_path))
        model.load_state_dict(torch.load(model_path))

        print('start from epoch {}'.format(start_epoch))
        max_epoch = args.epoches
        print ('max epoch', max_epoch)
    else: 
        max_epoch = args.epoches

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
   
    test_min_loss  = float('inf')
    import pandas as pd

    train_loss = np.array([])
    test_loss = np.array([])


    # save loss and compression ratio
    if not os.path.exists(os.path.join(checkpoints_path, 'results')):
        os.makedirs(os.path.join(checkpoints_path, 'results'))

    # load results if exists
    if os.path.exists(os.path.join(checkpoints_path, 'results', 'train_loss.csv')):
        train_loss = np.loadtxt(os.path.join(checkpoints_path, 'results', 'train_loss.csv'))
        test_loss = np.loadtxt(os.path.join(checkpoints_path, 'results', 'test_loss.csv'))

        print('load results from {}'.format(os.path.join(checkpoints_path, 'results')))
        print('start from epoch {}'.format(start_epoch))
        max_epoch = args.epoches
        print ('max epoch', max_epoch)

    for epoch in range(start_epoch, max_epoch):
        print('=' * 20, epoch + 1, '=' * 20)
        trainl = train_one_epoch(train_loader, model, loss_fn, optimizer)
        testl = test_one_epoch(test_loader, model, loss_fn)

        print("epoch: {}, train loss: {}, test loss: {}".format(epoch, trainl, testl))

        train_loss = np.append(train_loss, trainl)
        test_loss = np.append(test_loss, testl)
        
        saved_path = os.path.join(checkpoints_path, "model_{}.pth".format(epoch))
        torch.save(model.state_dict(), saved_path)
        np.savetxt(os.path.join(checkpoints_path, 'results', 'train_loss.csv'), train_loss)
        np.savetxt(os.path.join(checkpoints_path, 'results', 'test_loss.csv'), test_loss)


if __name__ == '__main__':
    main()