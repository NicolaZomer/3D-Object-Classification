import open3d as o3d
import os
import torch
import torch.nn as nn
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)


class encoder (nn.Module):
    def __init__ (self, npoints, ):
        super(encoder, self).__init__()
        self.npoints = npoints
        # self.batch_size = batch_size

        self.conv1 = self.conv2D(1, 64, [1, 3], padding='valid', bn=True, name='conv1')
        self.conv2 = self.conv2D(64, 64, [1, 1], bn=True, name='conv2')
        self.conv3 = self.conv2D(64, 64, [1, 1], bn=True, name='conv3')
        self.conv4 = self.conv2D(64, 128, [1, 1], bn=True, name='conv4')
        self.conv5 = self.conv2D(128, 1024, [1, 1], bn=True, name='conv5')
        self.maxpool = nn.MaxPool2d(kernel_size=[self.npoints, 1], padding=0, return_indices=True)

        self.fc1 = self.fully_connected(1024, 512, bn=False, name='fc1')


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


    def forward(self, x, batch_size):
        """ Classification PointNet, input is BxNx3, output Bx40 
        input is batch_size x nchannels x num_pointsx3 

        """
        verbose = False
        # unsqueeze to add channel dimension
        x = torch.unsqueeze(x, dim=1)# batch_size x 1 x num_points x 3 the second dimension is the channel dimension
        if verbose: print (x.shape)
        x = self.conv1(x)
        if verbose: print ('conv1', x.shape)
        x = self.conv2(x)
        if verbose: print ('conv2', x.shape)
        x = self.conv3(x)
        if verbose: print ('conv3', x.shape)
        x = self.conv4(x)
        if verbose: print ('conv4', x.shape)
        x = self.conv5(x)
        if verbose: print ('conv5', x.shape)
        # maxpool
        x, indices = self.maxpool(x)
        if verbose: print ('maxpool', x.shape)
        # FC layers
        x = torch.reshape (x, (batch_size, -1))
        if verbose: print ('reshape', x.shape)
        x = self.fc1(x)
        if verbose: print ('fc1', x.shape)
        
        return x, indices

class decoder (nn.Module):
    def __init__ (self, npoints):
        super(decoder, self).__init__()
        self.npoints = npoints

        # self.fc1 = self.fully_connected(64, 256, bn=False, name='fc1')
        # self.fc2 = self.fully_connected(256, 512, bn=False, name='fc2')
        self.fc3 = self.fully_connected(512, 1024, bn=False, name='fc3')

        self.maxunpool = nn.MaxUnpool2d(kernel_size=[self.npoints,1], padding=0)

        self.conv1 = self.unconv2D(1024, 512, 1, bn=False, name='conv1')
        self.conv2 = self.unconv2D(512, 256, 1, bn=False, name='conv2')
        self.conv3 = self.unconv2D(256, 256, 1, bn=False, name='conv3')
        self.conv4 = self.unconv2D(256, 128, 1, bn=False, name='conv4')
        self.conv5 = self.unconv2D(128, 1, [1,3], bn=False, name='conv5')

    def unconv2D(self, 
            num_in_channels,
            num_output_channels,
            kernel_size,
            stride=[1, 1],
            padding=0,
            activation_fn=nn.ReLU(),
            bn=False,
            name='conv2d'):
        """ 2D convolution with non-linear operation.

        """
        # create sequential model
        model = nn.Sequential()

        # add convolutional layer
        model.add_module(name+'conv', nn.ConvTranspose2d(num_in_channels, num_output_channels, kernel_size, stride, padding))
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
        if activation_fn is not None: model.add_module('activation_fn',activation_fn)

        return model

    def forward(self, x, indices, batch_size):
        """ Classification PointNet, input is BxNx3, output Bx40 
        input is batch_size x nchannels x num_pointsx3 

        """
        verbose = False
        if verbose: print ('decoder input', x.shape)
        x = self.fc3(x)
        if verbose: print ('fc3', x.shape)
        x = torch.reshape(x, [batch_size, 1024, 1, 1])

        #maxunpool2d
        x = self.maxunpool(x, indices)

        if verbose: print ('reshape', x.shape)
        x = self.conv1(x)
        if verbose: print ('conv1', x.shape)
        x = self.conv2(x)
        if verbose: print ('conv2', x.shape)
        x = self.conv3(x)
        if verbose: print ('conv3', x.shape)
        x = self.conv4(x)
        if verbose: print ('conv4', x.shape)
        x = self.conv5(x)
        if verbose: print ('conv5', x.shape)
        x = torch.reshape(x, [batch_size, self.npoints, 3])
        if verbose: print ('output', x.shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, npoints):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder(npoints)
        self.decoder = decoder(npoints)


    def forward(self, x, batch_size):
        x, indices = self.encoder(x, batch_size)
        encoded = x
        x = self.decoder(x, indices, batch_size)
        return x, encoded

if __name__ == '__main__':

    # load a real point
    import open3d as o3d

    path = '../dataset/ModelNet40/airplane/test/airplane_0627.off'
    x = o3d.io.read_triangle_mesh(path)
    x = np.asarray(x.vertices) 
    # get 10000 points  random
    num_points = 10000
    x = x[np.random.choice(x.shape[0], num_points, replace=False), :]


    # rehsape to num_pointsx3
    x = np.reshape(x, (-1, 3))

    # torch
    x = torch.from_numpy(x).float()
    # add batch dimension
    x = torch.unsqueeze(x, 0)
    print (x.shape)

    npoints = x.shape[1]
    batch_size = x.shape[0]
    model = AutoEncoder(npoints).float()

    # forward
    x, encoded = model(x, batch_size)
