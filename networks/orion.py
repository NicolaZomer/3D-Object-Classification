import torch
import torch.nn as nn
from collections import OrderedDict



class ORION(nn.Module):
    def __init__(self, input_shape, nclasses=40):
        super(ORION, self).__init__()
        self.n_classes = nclasses
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            OrderedDict([
            ('conv3d_1', nn.Conv3d(in_channels=1,out_channels=32, kernel_size=5)),
            ('relu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool3d(2)),
            ('drop1', nn.Dropout(p=0.5)),

            ('conv3d_2', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool3d(2)),
            ('drop2', nn.Dropout(p=0.6)),
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
    dataset = VoxelDataset(data_root, Train=True)
    data = dataset[0][0]
    print (data.shape)
    # unsqueeze to add batch dimension
    data = data.unsqueeze(0)

    voxnet = VoxNet(input_shape=(32, 32, 32))
    a = voxnet(data)
    print(a)

    # use loss function
    criterion = nn.CrossEntropyLoss()
    target = torch.LongTensor([1])
    loss = criterion(a, target)
    print(loss)