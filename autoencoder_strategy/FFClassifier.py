"""
just a simple feed forward classifier for testing

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class FFClassifier(nn.Module):
    def __init__(self, input_dim=512, nclasses=40):
        super(FFClassifier, self).__init__()
        #lay1
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.7)

        #lay2
        self.fc2 = nn.Linear(256, 128)


        #lay2
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)       
        self.dropout2 = nn.Dropout(p=0.7)

        # lay3
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(p=0.7)
        self.fc4 = nn.Linear(64, nclasses)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x


class classifier ():
    def __init__(self, input_dim=512, nclasses=40, lr=1e-4, weight_decay=0.0001):

        self.model = FFClassifier(input_dim=input_dim, nclasses=nclasses)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_dir = 'FFClassifier_checkpoint'

        self.train_loss = []
        self.val_loss = []

    def fit(self, train_dataloader, val_loader, epochs=30):

        min_val_loss = np.inf
        for epoch_num in tqdm(range(epochs)):
            # print(f'EPOCH {epoch_num}')

            ### TRAIN
            self.model.train() # Training mode (e.g. enable dropout, batchnorm updates,...)
            for sample_batched in train_dataloader:
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)
                # print (x_batch.shape, label_batch.shape)
    
                out = self.model(x_batch)
                # Compute loss
                loss = self.loss(out, label_batch)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_loss.append(loss.item())

            ### VALIDATION
            
            self.model.eval() # Evaluation mode (e.g. disable dropout, batchnorm updates,...)
            # print ("TESTING")
            with torch.no_grad(): # Disable gradient tracking
                for sample_batched in val_loader:
                    # Move data to device
                    x_batch = sample_batched[0].to(self.device)
                    label_batch = sample_batched[1].to(self.device)

                    # Forward pass
                    out = self.model(x_batch)

                    # Compute loss cross entropy
                    loss = self.loss(out, label_batch)
                    self.val_loss.append(loss.item())

                    # save the best model
                    if loss.item() < min_val_loss:
                        min_val_loss = loss.item()
                        self.save(self.save_dir, 'best.pth')

        # save the model
        self.save(self.save_dir, 'best.pth')

        # save the loss
        np.savetxt(os.path.join(self.save_dir, 'train_loss.txt'), np.array(self.train_loss))
        np.savetxt(os.path.join(self.save_dir, 'val_loss.txt'), np.array(self.val_loss))



    def test(self, test_dataloader):
        self.model.eval()
        with torch.no_grad():
            for sample_batched in tqdm(test_dataloader):
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                label_batch = sample_batched[1].to(self.device)

                # Forward pass
                out = self.model(x_batch)
                # Compute loss cross entropy
                loss = self.loss(out, label_batch)
                self.val_loss.append(loss.item())

    def predict(self, test_dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for sample_batched in tqdm(test_dataloader):
                # Move data to device
                x_batch = sample_batched[0].to(self.device)
                # Forward pass
                out = self.model(x_batch)
                # apply softmax
                out = F.softmax(out, dim=1) 
                # get the argmax
                pred = torch.argmax(out, dim=1)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)

    def save(self, path, name):
        import os
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), os.path.join(path, name))

    def load(self, name):
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, name)))