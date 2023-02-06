import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import torch 
import torch.nn as nn
import torch.nn.functional as F

from custom_losses import TripletCenterLoss


class TripletNet (nn.Module):
    
    def __init__ (self, nclasses, num_point=10000):
        super(TripletNet, self).__init__()
        self.num_classes = nclasses
        
        self.loss = TripletCenterLoss()

        self.conv1 = self.conv2D(1, 64, [1, 3], padding='valid', bn=True, name='conv1')
        self.conv2 = self.conv2D(64, 64, [1, 1], bn=True, name='conv2')
        self.conv3 = self.conv2D(64, 64, [1, 1], bn=True, name='conv3')
        self.conv4 = self.conv2D(64, 128, [1, 1], bn=True, name='conv4')
        self.conv5 = self.conv2D(128, 1024, [1, 1], bn=True, name='conv5')


        self.fc1 = self.fully_connected(1024, 512, bn=False, name='fc1')
        
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = self.fully_connected(512, 256, bn=False, name='fc2')
        
                
        #self.fc3 = self.fully_connected(256, self.num_classes, activation_fn=None, bn=False, name='fc3')
        
        #self.custlayer = self.norm_l2_layer(self.fc3)
        

    def conv2D(self, 
            num_in_channels,
            num_output_channels,
            kernel_size,
            stride=[1, 1],
            padding='same',
            activation_fn=nn.ReLU(),
            bn=False,
            name='conv2d'):
        """ 2D convolution with non-linear operation.

        """
        # create sequential model
        model = nn.Sequential()

        # add convolutional layer
        model.add_module(name+'conv', nn.Conv2d(num_in_channels, num_output_channels, kernel_size, stride, padding))
        if bn: model.add_module(name+'batch_norm', nn.BatchNorm2d(num_output_channels))
        if activation_fn is not None: model.add_module('activation_fn', activation_fn)

        return model

    def fully_connected(self,
            input_size,
            output_size,
            activation_fn=nn.ReLU(),
            bn=False,
            name='fully_connected'):
        """ Fully connected layer with non-linear operation.

        """

        # create sequential model
        model = nn.Sequential()

        # add fully connected layer
        model.add_module(name+'fc', nn.Linear(input_size, output_size))
        if bn: model.add_module(name+'batch_norm', nn.BatchNorm1d(output_size))
        if activation_fn is not None: model.add_module('activation_fn', activation_fn)

        return model

    
    
    def get_loss(self, input, output):

        return self.loss(input, output)


    def forward(self, x):
        """ Classification TripletNet, input is BxNx3, output Bx40 
        input is batch_size x nchannels x num_pointsx3 

        """
        verbose = False

        num_point = x.shape[1] # number of points
        batch_size = x.shape[0] # batch size

        num_in_channels = 1 # number of input channels

        if verbose:
            print ('num_point: ', num_point)
            print ('batch_size: ', batch_size)
            print ('num_in_channels: ', num_in_channels)

        # unsqueeze to add channel dimension
        x = torch.unsqueeze(x, dim=1)# batch_size x 1 x num_points x 3 the second dimension is the channel dimension

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

    
        # Symmetric function: max pooling
        x = nn.MaxPool2d(kernel_size=[num_point, 1], padding=0)(x)

        # FC layers
        x = torch.reshape(x, [batch_size, -1])
        x = self.fc1(x)
         # dropout
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        
        #x = self.fc3(x)
        
        x = F.normalize(x, p=2, dim=1)

        return x

if __name__ == '__main__':

    # test
    # batch_size = 32
    # num_point = 1024
    # num_channel = 3
    # x = torch.rand(batch_size, num_point, 3) # batch_size x nchannels x num_pointsx3

    # load a real point
    import open3d as o3d

    path = '../dataset/ModelNet40/airplane/test/airplane_0627.off'
    x = o3d.io.read_triangle_mesh(path)
    x = np.asarray(x.vertices)
    # rehsape to num_pointsx3
    x = np.reshape(x, (-1, 3))
    print (x.shape)

    # torch
    x = torch.from_numpy(x).float()
    # add batch dimension
    x = torch.unsqueeze(x, 0)
    print (x.shape)

    model = TripletNet(nclasses=40).float()

    # print parameters
    for name, param in model.named_parameters():
        print (name, param.shape)

    # print net
    print (model)

    output = model(x)
    print(output)

    # check normalisation
    print (torch.sum(output, dim=1))

    # check prediction
    print (torch.argmax(output, dim=1))
