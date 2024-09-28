import torch
import torch.nn as nn
from .layers_ours import Linear, AdaptiveAvgPool1d, Softmax


class mlp(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96):
        super(mlp, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        # num_outputs = num_classes
        self.hidden = Linear(num_tokens, 4*num_tokens)
        self.head = Linear(4*num_tokens, num_outputs)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        # x -> (b, 96, 4, 4, 4, t)
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = self.head(x)
        # x = self.softmax(x)
        return x
    
    def relprop(self, cam, **kwargs):
        # cam = self.softmax.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = self.head.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        print("MLP v2 size", cam.size)
        cam = self.hidden.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = cam.view(cam.size(0), cam.size(1), 1)  # Reshape to match output shape of avgpool (B, C, 1)
        cam = cam.transpose(1, 2)
        cam = self.avgpool.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = cam.transpose(1, 2)
        cam = cam.reshape(cam.size(0), -1, 4, 4, 4, cam.size(-1))  # Undo flattening back to (b, 96, 4, 4, 4, t)
        return cam
