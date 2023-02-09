import torch
import torch.nn as nn
from collections import OrderedDict



class VoxNet(nn.Module):
    def __init__(self, input_shape, nclasses=40):
        super(VoxNet, self).__init__()
        self.n_classes = nclasses
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            OrderedDict([
            ('conv3d_1', nn.Conv3d(in_channels=1,out_channels=32, kernel_size=5, stride=2)),
            ('relu1', nn.LeakyReLU()),
            ('drop1', nn.Dropout(p=0.5)),

            ('conv3d_2', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool3d(2)),
            ('drop2', nn.Dropout(p=0.6)),

            ('conv3d_3', nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)),
            ('relu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool3d(2)),
            ('drop3', nn.Dropout(p=0.7)),

        ]))
        x = self.conv(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = nn.Sequential(OrderedDict([

            ('fc1', nn.Linear(dim_feat, 128)),
            ('relu1', nn.LeakyReLU()),
            ('drop3', nn.Dropout(p=0.4)),

            ('fc2', nn.Linear(128, self.n_classes)),
        
        ]))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from dataset.voxelDataset import VoxelDataset

    data_root = '../dataset/ModelNet40'
    dataset = VoxelDataset(data_root, train=True)
    data = dataset[0][0]
    print (data.shape)
    # unsqueeze to add batch dimension
    data = data.unsqueeze(0)

    voxnet = VoxNet(input_shape=(32, 32, 32))
    # model = voxnet(data)
    sys.path.append('../')
    from utils import count_parameters
    
    count_parameters(voxnet, data)