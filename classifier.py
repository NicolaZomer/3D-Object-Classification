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
            optimizer,
            save_dir='checkpoints', 
            start_epoch=0
            ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loss_log = []
        self.val_loss_log = []

        # if save_dir doews not exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # load loss log
        if os.path.exists(f'{save_dir}/train_loss.npy'):
            train_loss = np.load(f'{save_dir}/train_loss.npy')
            self.train_loss_log = train_loss.tolist()

        if os.path.exists(f'{save_dir}/val_loss.npy'):
            val_loss = np.load(f'{save_dir}/val_loss.npy')
            self.val_loss_log = val_loss.tolist()

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

            
            np.save(f'{save_dir}/train_loss.npy', self.train_loss_log)
            np.save(f'{save_dir}/val_loss.npy', self.val_loss_log)

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