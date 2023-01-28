import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class Classifier ():

    def __init__(self, model = None, device = 'cpu'):
        torch.manual_seed(0)
        self.device = device

        self.Net = model
        self.Net.to(device)

    def train(self, train_dataloader, val_dataloader, loss_fn, epochs, lr=1e-3, save_dir='checkpoints'):
        parameters = []
        for name, param in self.Net.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        print ('parameters', parameters)
        self.optimizer =  optim.Adam(parameters, lr=lr)
        self.loss_fn = loss_fn

        self.train_loss_log = []
        self.val_loss_log = []

        for epoch_num in range(epochs):
            if epoch_num % 200 == 0:
                print('*'*50)
                print(f'EPOCH {epoch_num}')

            ### TRAIN
            train_loss= []
            self.Net.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            for sample_batched in train_dataloader:
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)

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

            val_loss= []
            self.Net.eval() # Evaluation mode (e.g. disable dropout, batchnorm,...)

            with torch.no_grad(): # Disable gradient tracking
                for sample_batched in val_dataloader:
                    # Move data to device
                    x_batch = sample_batched[0].to(self.device)
                    label_batch = sample_batched[1].to(self.device)

                    # Forward pass
                    out = self.Net(x_batch)

                    # Compute loss
                    loss = self.loss_fn(out, label_batch)

                    # Save val loss for this batch
                    loss_batch = loss.detach().cpu().numpy()
                    val_loss.append(loss_batch)


                # Save average validation loss
                val_loss = np.mean(val_loss)
                self.val_loss_log.append(val_loss)

            print(f"Epoch {epoch_num} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
            # save model
            self.save_state_dict(f'{save_dir}/model_{epoch_num}.torch')
            self.save_optimizer_state(f'{save_dir}/optimizer_{epoch_num}.torch')

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

        all_output_classes = torch.zeros(all_outputs.shape).to(self.device)
        all_output_classes[all_outputs > 0] = 1 # output greater than 0 is class 1
        tot_correct_out = (all_output_classes == all_labels).sum()
        test_accuracy =  tot_correct_out / len(all_labels)
        print(f"TEST ACCURACY: {test_accuracy:.2f}%")

        return test_accuracy

    
    def test (self, test_dataloader, device, plot=True):
        all_inputs = []
        all_outputs = []
        all_labels = []
        self.Net.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # Disable gradient tracking
            for sample_batched in test_dataloader:
                # Move data to device
                x_batch = sample_batched[0].to(device)
                label_batch = sample_batched[1].to(device)
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

        test_loss = self.loss_fn(all_outputs, all_labels)
        test_acc = self._accuracy(all_outputs, all_labels)
        print(f"AVERAGE TEST LOSS: {test_loss}")
        predictions = all_outputs.detach().cpu().numpy()

        return test_loss, test_acc, predictions



if __name__ == '__main__':
    from dataset import *
    from networks.PointNet import PointNet

    dataset_train  = PointCloudDataset('dataset/modelnet40', train=True)
    dataset_val    = PointCloudDataset('dataset/modelnet40', train=False)
    print (f"Train dataset size: {len(dataset_train)}")
    print (f"Val dataset size: {len(dataset_val)}")


    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    dataloader_val   = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Net = PointNet(nclasses=40)
    Net.to(device)

    # show the network architecture
    print(Net)
    

    classifier = Classifier(Net, device=device)
    loss_fn = nn.CrossEntropyLoss()
    classifier.train(dataloader_train, dataloader_val, epochs=10, lr=0.001, loss_fn=loss_fn, save_dir='checkpoints/pointnet')

    classifier.plot_history()


    




