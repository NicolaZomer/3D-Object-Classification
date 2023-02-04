import torch
import torch.nn as nn
from collections import OrderedDict



class ResVoxNet(nn.Module):

    def __init__(self, input_shape, nclasses=40):
        super(ResVoxNet, self).__init__()
        self.n_classes = nclasses
        self.input_shape = input_shape

        self.conv1 = self.conv_layer(inchannel=1, outchannel=128, kernel_size=3, stride=2, padding=0)
        #pool
        self.pool = nn.MaxPool3d(2)

        self.conv2 = self.conv_layer(inchannel=128, outchannel=128, kernel_size=1, stride=1, padding=0)
        self.conv3 = self.conv_layer(inchannel=128, outchannel=128, kernel_size=1, stride=1, padding=0)
        self.conv4 = self.conv_layer(inchannel=128, outchannel=128, kernel_size=1, stride=1, padding=0)

        # compute the dimension of the feature vector with a forward pass
        x = torch.autograd.Variable(torch.rand((1, 1) + input_shape))
        x = self.conv1(x) 
        # print (f"conv1: {x.size()}")
        x = self.pool(x)
        # print (f"pool: {x.size()}")
        # print (f"conv2: {self.conv2(x).size()}")
        x = self.conv2(x) + x
        x = self.conv3(x) + x
        x = self.conv4(x) + x

        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(dim_feat, 64)),
            ('relu1', nn.LeakyReLU()),
            ('drop3', nn.Dropout(p=0.6)),
            ('fc2', nn.Linear(64, self.n_classes)),
        
        ]))

    def conv_layer (self, inchannel, outchannel, kernel_size, stride, padding):
        model = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel_size, 
                        stride=stride, padding=padding),
            nn.BatchNorm3d(outchannel),
            nn.Dropout(p=0.6)
        )
        return model


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x) + x
        x = nn.ReLU()(x)
        x = self.conv3(x) + x
        x = nn.ReLU()(x)
        x = self.conv4(x) + x
        x = nn.ReLU()(x)

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

    voxnet = ResVoxNet(input_shape=(32, 32, 32))
    a = voxnet(data)
    print(a)

    # use loss function
    criterion = nn.CrossEntropyLoss()
    target = torch.LongTensor([1])
    loss = criterion(a, target)
    print(loss)