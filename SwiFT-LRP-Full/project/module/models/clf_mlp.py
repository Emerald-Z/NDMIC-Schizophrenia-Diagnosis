import torch
import torch.nn as nn
from .value_opt_layers import Linear, Softmax, Dropout
from .layers_ours import AdaptiveAvgPool1d


class mlp(nn.Module):
    def __init__(self, num_classes=2, num_tokens = 96, labels={}):
        super(mlp, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes
        self.head = Linear(in_features=num_tokens, out_features=num_outputs, labels=labels["l"])
        self.avgpool = AdaptiveAvgPool1d(1)
        self.softmax = Softmax(dim=1, labels=labels["sm"])
        self.dropout = Dropout(p=0.8, labels=labels["do"])

    def forward(self, x):
        # x -> (1, 288, 2, 2, 2, 20)
        x = x.flatten(start_dim=2).transpose(1, 2) 
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.head.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = cam.view(cam.size(0), cam.size(1), 1)  # Reshape to match output shape of avgpool (B, C, 1)
        cam = self.avgpool.relprop(cam, alpha=kwargs["alpha"]) 
        cam = cam.reshape(cam.size(0), cam.size(1), 2, 2, 2, -1)  # Undo flattening back to (b, 96, 4, 4, 4, t)
        return cam
