import torch
from torch import nn

flops = 0 
def count_flops(m, i, o): 
    global flops 
    x = i[0] 
    flops += (2 * x.nelement() - 1) * m.weight.nelement() 

def count_parameters(model, data):
    hooks = [] 

    for name, module in model.named_modules(): 
        if isinstance(module,nn.Sequential):
            for name, layer in module._modules.items():
                if isinstance(layer, nn.Conv3d): 
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.Linear): 
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.ConvTranspose3d): 
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.BatchNorm3d): 
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.Conv2d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.ConvTranspose2d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.Conv1d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.ConvTranspose1d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.BatchNorm1d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.BatchNorm2d):
                    hooks.append(layer.register_forward_hook(count_flops))
                if isinstance(layer, nn.BatchNorm3d):
                    hooks.append(layer.register_forward_hook(count_flops))


        else:
            if isinstance(module, nn.Conv3d): 
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.Linear): 
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.ConvTranspose3d): 
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.ConvTranspose2d):
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.Conv1d):
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.ConvTranspose1d):
                hooks.append(module.register_forward_hook(count_flops))

            if isinstance(module, nn.BatchNorm1d):
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(count_flops))
            if isinstance(module, nn.BatchNorm3d): 
                hooks.append(module.register_forward_hook(count_flops))
            


    with torch.no_grad(): 
        model(data) 

    for hook in hooks: 
        hook.remove() 

    print(f"FLOPs: {flops/10**9}")

    



    
