from .value_opt_layers import Linear, safe_divide
from unittest.mock import patch
from torch import nn as nn
import torch

class PatchEmbed(nn.Module):
    """ 4D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size=(96, 96, 96, 20),
        patch_size=(6, 6, 6, 1),
        in_chans=2,
        embed_dim=24,
        norm_layer=None,
        flatten=True,
        spatial_dims=3,
        labels=[]
    ):
        print(patch_size)
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
        assert patch_size[3] == 1, "temporal axis merging is not implemented yet"

        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size # TODO: change this back
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            # img_size[3] // patch_size[3],
        )
        print(self.grid_size)
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # FC layer to turn patches into embeddings
        self.fc = Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3], out_features=embed_dim, labels=labels["l"])

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D, T = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        assert D == self.img_size[2], f"Input image width ({D}) doesn't match model ({self.img_size[2]})."
        x = self.proj(x)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def proj(self, x):
        B, C, H, W, D, T = x.shape
        pH, pW, pD = self.grid_size
        sH, sW, sD, sT = self.patch_size

        x = x.view(B, C, pH, sH, pW, sW, pD, sD, -1, sT)
        x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous()
        x = x.view(-1, sH * sW * sD * sT * C)
        x = self.fc(x.float())
        x = x.view(B, pH, pW, pD, -1, self.embed_dim).contiguous()
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x

    def relprop(self, cam, **kwargs):  
        # Undo flattening and projection
        if self.flatten:
            cam = cam.transpose(1, 2).contiguous().view(-1, self.embed_dim, *self.grid_size) # BNC -> BCHW
        cam = cam.reshape((-1, self.embed_dim))
        cam = self.fc.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = cam.view(1, self.grid_size[0], self.grid_size[1], self.grid_size[2], -1, *self.patch_size, 1) # self.fc.in_features // self.num_patches
        cam = cam.permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
        cam = cam.view(-1, self.img_size[0], self.img_size[1], self.img_size[2], self.img_size[3])
        return cam 
