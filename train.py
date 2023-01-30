import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import  recall_score, precision_score, f1_score, accuracy_score

from dataset.PointCloudDataset import PointCloudDataset
from dataset.voxelDataset import VoxelDataset

from networks.PointNet import PointNet
from networks.voxnet import VoxNet

import torch.nn as nn

class Classifier ():
    def __init__(self, model = None, device = 'cpu'):
        torch.manual_seed(0)
        self.device = device

        self.Net = model
        self.Net.to(device)

    def train(self,
            train_dataloader, 
            val_dataloader, 
            loss_fn, 
            epochs, 
            lr=1e-3, 
            weight_decay=1e-5,
            save_dir='checkpoints', 
            start_epoch=0
            ):
        parameters = self.Net.parameters()
        # self.optimizer =  optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        self.optimizer =  optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn

        self.train_loss_log = []
        self.val_loss_log = []

        # if save_dir doews not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch_num in range(start_epoch, start_epoch+epochs):
            print(f'EPOCH {epoch_num}')

            ### TRAIN
            train_loss= []
            self.Net.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            print ("TRAINING")
            for sample_batched in tqdm(train_dataloader):
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)

                # print (x_batch.shape)
                # print (label_batch.shape)

                out = self.Net(x_batch)

                # Compute loss
                loss = self.loss_fn(out, label_batch)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Save train loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)
        
            # Save average train loss
            train_loss = np.mean(train_loss)
            self.train_loss_log.append(train_loss)

            # Validation
            val_loss= []
            self.Net.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)

            with torch.no_grad(): # Disable gradient tracking
                print ("TESTING")
                for sample_batched in tqdm(val_dataloader):
                    # Move data to device
                    x_batch = sample_batched[0].to(self.device)
                    label_batch = sample_batched[1].to(self.device)

                    # Forward pass
                    out = self.Net(x_batch)

                    # Compute loss cross entropy
                    loss = self.loss_fn(out, label_batch)

                    # Save val loss for this batch
                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(loss_batch)

                # Save average validation loss
                val_loss = np.mean(val_loss)
                self.val_loss_log.append(val_loss)

            # logs
            print(f"Epoch {epoch_num} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            # save model
            self.save_state_dict(f'{save_dir}/model_{epoch_num}.torch')
            self.save_optimizer_state(f'{save_dir}/optimizer_{epoch_num}.torch')

            # save loss, add if file exists
            if os.path.exists(f'{save_dir}/train_loss.npy'):
                train_loss = np.load(f'{save_dir}/train_loss.npy')
                train_loss = np.append(train_loss, self.train_loss_log)
                np.save(f'{save_dir}/train_loss.npy', train_loss)
            else: np.save(f'{save_dir}/train_loss.npy', self.train_loss_log)

            if os.path.exists(f'{save_dir}/val_loss.npy'):
                val_loss = np.load(f'{save_dir}/val_loss.npy')
                val_loss = np.append(val_loss, self.val_loss_log)
                np.save(f'{save_dir}/val_loss.npy', val_loss)
            else: np.save(f'{save_dir}/val_loss.npy', self.val_loss_log)

    def history(self):
        return self.train_loss_log, self.val_loss_log
    
    def plot_history(self, save_dir='.'):
        import seaborn as sns
        sns.set_theme (style="darkgrid", font_scale=1.5, rc={"lines.linewidth": 2.5, "lines.markersize": 10})
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_log, label='train')
        plt.plot(self.val_loss_log, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    
    def predict(self, x, numpy=False):
        self.Net.eval()
        with torch.no_grad(): # turn off gradients computation
            out = self.Net(x)
            # compute prob
            out = torch.nn.functional.softmax(out, dim=1)
            # get the class
            out = torch.argmax(out, dim=1)

        print(f"Output shape: {out.shape}")
        if numpy:
            out = out.detach().cpu().numpy()

        else:
            return out
        
    def get_weights(self, numpy=True):
        dict_weights = {}
        names = self.Net.state_dict().keys()
        print (names)
        if not numpy:
            for name in names:
                dict_weights[name] = self.Net.state_dict()[name]
        else:
            for name in names:
                dict_weights[name] = self.Net.state_dict()[name].detach().cpu().numpy()

        return dict_weights

    def save_state_dict(self, path):

        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        print (f"Saving model to {path}")
        net_state_dict = self.Net.state_dict()
        torch.save(net_state_dict, path)

    def load_state_dict(self, path):
        
        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        
        # check if path exists
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        net_state_dict = torch.load(path)
        # Update the network parameters
        self.Net.load_state_dict(net_state_dict)

    def save_optimizer_state(self, path):

        if path.split('.')[-1] != 'torch':
            path = path + '.torch'

        ### Save the self.optimizer state
        torch.save(self.optimizer.state_dict(), path)

    def load_optimizer_state(self, path):
        if path.split('.')[-1] != 'torch':
            path = path + '.torch'
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        ### Reload the optimizer state
        opt_state_dict = torch.load(path)
        self.optimizer.load_state_dict(opt_state_dict)

    def _accuracy (self,all_outputs, all_labels):
        # the output doesnt do softmax, so we need to do it
        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_output_classes = torch.argmax(probs, dim=1)
    
        # compute accuracy
        test_accuracy = accuracy_score(all_labels, all_output_classes)
        print(f"TEST ACCURACY: {test_accuracy:.2f}%")

        return test_accuracy

    def _recall_precision (self, all_outputs, all_labels):
        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_output_classes = torch.argmax(probs, dim=1)

        recall = recall_score(all_labels, all_output_classes, average='macro')
        precision = precision_score(all_labels, all_output_classes, average='macro')

        print(f"TEST RECALL: {recall:.2f}%")
        print(f"TEST PRECISION: {precision:.2f}%")

        return recall, precision


    def test (self, test_dataloader):
        all_inputs = []
        all_outputs = []
        all_labels = []
        self.Net.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # Disable gradient tracking
            for sample_batched in tqdm(test_dataloader):
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)
                # Forward pass
                out = self.Net(x_batch)
                # Save outputs and labels
                all_inputs.append(x_batch)
                all_outputs.append(out)
                all_labels.append(label_batch)
        # Concatenate all the outputs and labels in a single tensor
        all_inputs  = torch.cat(all_inputs)
        all_outputs = torch.cat(all_outputs)
        all_labels  = torch.cat(all_labels)

        probs = torch.nn.functional.softmax(all_outputs, dim=1)
        all_predictions = torch.argmax(probs, dim=1)

        test_acc = self._accuracy(all_outputs, all_labels)

        recall = recall_score(all_labels, all_predictions, average='macro') 
        precision = precision_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        results = pd.DataFrame()
        results['accuracy'] = [test_acc]
        results['recall'] = [recall]
        results['precision'] = [precision]
        results['f1'] = [f1]
        return results


import argparse
import os
# 2000 dati allenati con batch 32 e 5e-4
# 4000 dati allenati con batch 64 e 1e-3

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--model_name', type=str, default='voxnet', help='model name (default: pointnet)', choices=['pointnet', 'voxnet'])
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 32)')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints (default: checkpoints/pointnet)')
parser.add_argument('--ndata', type=int, default=100, help='number of data points to use (default: 2000)')
parser.add_argument('--npoints', type=int, default=2000, help='number of points in the point cloud (default: 1024)')
parser.add_argument('--train', type=bool, default=True, help='train or test (default: True)')
args = parser.parse_args()


def main (
    model_name='pointnet',
    epochs=10,
    lr=0.001,
    batch_size=32,
    save_dir='checkpoints',
    ndata=4000,
    npoints = 1024,
    train=False,
    ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(save_dir, model_name)

    ##############################
    # LOAD MODEL AND DATASET
    ##############################
    print (f"Loading model {model_name}...")
    if model_name == 'pointnet':

        dataset_train  = PointCloudDataset('dataset/modelnet40_normal_resampled', 
                                            train=True, 
                                            ndata=ndata, 
                                            file_extension='.txt', 
                                            npoints=npoints
                                        )
        if train: test_data = int(ndata/20)
        else: test_data = -1
        dataset_val    = PointCloudDataset('dataset/modelnet40_normal_resampled', 
                                            train=False, 
                                            ndata=test_data,
                                            file_extension='.txt', 
                                            npoints=npoints
                                        )

        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")
    
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

        Net = PointNet(nclasses=40)


    elif model_name == 'voxnet':
        input_shape = (32, 32, 32)
        dataset_train  = VoxelDataset('dataset/ModelNet40', 
                                        train=True, 
                                        ndata=ndata, 
                                        file_extension='.off', 
                                        )
        if train: test_data = int(ndata/20)
        else: test_data = -1
        dataset_val    = PointCloudDataset('dataset/ModelNet40', 
                                            train=False, 
                                            ndata=test_data,
                                            file_extension='.off', 
                                        )

        print (f"Train dataset size: {len(dataset_train)}")
        print (f"Val dataset size: {len(dataset_val)}")

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)

        print (f"first element in loader: {next(iter(dataloader_train))}")
        # print number of elments in loader

        Net = VoxNet(input_shape=input_shape, nclasses=40)
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
    loss_fn = nn.CrossEntropyLoss()

    if train:
        print ("Starting training")
        classifier.train(dataloader_train, dataloader_val, epochs=epochs, lr=lr, loss_fn=loss_fn, save_dir=save_dir, start_epoch=epoches_done+1)
        print ("Training done")
    else:
        print ("Starting testing")
        results = classifier.test(dataloader_val)
        print ("Testing done")
        print(results)


if __name__ == '__main__':
    main(
       **vars(args)
    )