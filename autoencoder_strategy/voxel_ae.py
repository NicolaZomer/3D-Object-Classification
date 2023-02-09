"""
https://arxiv.org/pdf/2202.10099.pdf idea
"""
import torch
import torch.nn as nn
import torch.optim as optim

class voxEncoder (nn.Module):
    def __init__(self, input_shape=(32, 32, 32), latent_dim=512):
        super(voxEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, out_channels=32, kernel_size=7)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout3d(p=0.5)

        self.conv2 = nn.Conv3d(32, out_channels=16, kernel_size=7)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout3d(p=0.5)

        self.conv3 = nn.Conv3d(16, out_channels=8, kernel_size=7)
        self.bn3 = nn.BatchNorm3d(8)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout3d(p=0.5)

        x = torch.randn(input_shape).view(-1, 1, *input_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop3(x)

        self.flatten = nn.Flatten()
        in_dim = x[0].shape[0] * x[0].shape[1] * x[0].shape[2] * x[0].shape[3]
        self.fc1 = nn.Linear(in_dim, latent_dim)


    def forward(self, x):
        x = self.conv1(x)
        # print ('conv1', x.shape)
        # print ('maxpool1', x.shape)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.relu1 (x)
        # print ('drop1', x.shape)

        x = self.conv2(x)
        # print ('conv2', x.shape)
        # print ('maxpool2', x.shape)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.relu2 (x)
        # print ('drop2', x.shape)

        x = self.conv3(x)
        # print ('conv3', x.shape)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.relu3 (x)
        # print ('drop3', x.shape)

        x = self.flatten(x)
        # print ('flatten', x.shape)
        x = self.fc1(x)
        # print ('fc1', x.shape)
        return x

class voxDecoder (nn.Module):
    def __init__(self, input_shape=(32, 32, 32), latent_dim=512):
        super(voxDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 21952)
        self.unflatten = nn.Unflatten(1, (8, 14, 14, 14))
    
        self.conv1 = nn.ConvTranspose3d(8, out_channels=16, kernel_size=7)
        self.drop1 = nn.Dropout3d(p=0.5)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose3d(16, out_channels=32, kernel_size=7)
        self.drop2 = nn.Dropout3d(p=0.5)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose3d(32, out_channels=1, kernel_size=7)
        self.drop3 = nn.Dropout3d(p=0.5)
        self.bn3 = nn.BatchNorm3d(1)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        # print ('fc1', x.shape)
        x = self.unflatten(x)
        # print ('unflatten', x.shape)

        x = self.conv1(x)
        # print ('conv1', x.shape)
        x = self.drop1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # print ('drop1', x.shape)

        x = self.conv2(x)
        # print ('conv2', x.shape)
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # print ('drop2', x.shape)

        x = self.conv3(x)
        # print ('conv3', x.shape)
        x = self.drop3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # print ('drop3', x.shape)

        return x


class voxAutoEncoder (nn.Module):

    def __init__(self, input_shape=(32, 32, 32), latent_dim=512):
        super(voxAutoEncoder, self).__init__()
        self.encoder = voxEncoder(input_shape, latent_dim)
        self.decoder = voxDecoder(input_shape, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        encoded = x
        decoded = self.decoder(x)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)


if __name__ == '__main__':

    data = torch.randn((1, 1, 32, 32, 32))
    model = voxAutoEncoder()

    flops = 0 
    hooks = [] 
    
    def count_flops(m, i, o): 
        global flops 
        x = i[0] 
        flops += (2 * x.nelement() - 1) * m.weight.nelement() 
    
    for name, module in model.named_modules(): 
        if isinstance(module, nn.Conv3d): 
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.Linear): 
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.ConvTranspose3d): 
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.BatchNorm3d): 
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.ConvTranspose2d):
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.MaxPool2d):
            hooks.append(module.register_forward_hook(count_flops))
        if isinstance(module, nn.MaxPool3d):
            hooks.append(module.register_forward_hook(count_flops))

    
    with torch.no_grad(): 
        model(data) 
    for hook in hooks: 
        hook.remove() 
    print(f"FLOPs: {flops/10**9}")

    



    
