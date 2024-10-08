"""
Our code is based on the following code.
https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR
"""

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
# from torch.nn import LayerNorm

# from monai.networks.blocks import MLPBlock as Mlp
from .mlpblock import MLPBlock as Mlp

from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .patchembedding import PatchEmbed
from .value_opt_layers import *

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "window_partition",
    "window_reverse",
    "WindowAttention4D",
    "SwinTransformerBlock4D",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer4D",
]


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, D, H, W, T, C)

        window_size: local window size.


    Returns:
        windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()

    b, d, h, w, t, c = x_shape
    x = x.view(
        b,
        d // window_size[0],  # number of windows in depth dimension
        window_size[0],  # window size in depth dimension
        h // window_size[1],  # number of windows in height dimension
        window_size[1],  # window size in height dimension
        w // window_size[2],  # number of windows in width dimension
        window_size[2],  # window size in width dimension
        t // window_size[3],  # number of windows in time dimension
        window_size[3],  # window size in time dimension
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
        .contiguous()
        .view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], c)
    )
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor (B*num_windows, window_size, window_size, C)
        window_size: local window size.
        dims: dimension values.

    Returns:
        x: (B, D, H, W, T, C)
    """
    b, d, h, w, t = dims
    x = windows.view(
        b,
        torch.div(d, window_size[0], rounding_mode="floor"),
        torch.div(h, window_size[1], rounding_mode="floor"),
        torch.div(w, window_size[2], rounding_mode="floor"),
        torch.div(t, window_size[3], rounding_mode="floor"),
        window_size[0],
        window_size[1],
        window_size[2],
        window_size[3],
        -1,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(b, d, h, w, t, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]

    # recursively multiply layer attentions: from paper def of rollout
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


'''
    these save_ functions provide a mechanism for capturing and storing relevant intermediate values during the forward pass 
    of the attention mechanism. These values are then used during the relevance propagation step to interpret the model's predictions 
    and understand the importance of input features
'''

class WindowAttention4D(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        labels: dict = {}
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__
        self.qkv = Linear(in_features=dim, out_features=dim * 3, labels=labels["l1"], bias=qkv_bias) # TODO: for some reason need to label params?
        self.attn_drop = Dropout(p=attn_drop, labels=labels["do"])
        self.proj = Linear(in_features=dim, out_features=dim, labels=labels["l2"])
        self.proj_drop = Dropout(p=proj_drop, labels=labels["do"])
        self.softmax = Softmax(dim=-1, labels=labels["sm"])

        self.add = Add(labels=labels["add"])

        # A = Q*K^T
        self.matmul1 = einsum('bhij,bhjd->bhid', labels=labels["ein1"]) # bhid,bhjd->bhij
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid', labels=labels["ein2"]) # what are the inputs TODO: check this help

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    '''
        From Translrp
    '''
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients
    
    def hook_test(self, grad):
        print("simple hook test. grad shape:", grad.shape)
        
    def forward(self, x, mask):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        b_, n, c = x.shape
        # 3 matrices
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        # b_, heads, n, c//heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        self.save_v(v)
        # q = q * self.scale
        # b_, heads, 32, 32 token x token
        attn = self.matmul1([q, k.transpose(-2, -1)]) 
        if mask is not None:
            nw = mask.shape[0]
            attn = self.add([attn.view(b_ // nw, nw, self.num_heads, n, n), mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)])
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        self.save_attn(attn)
        if self.train and x.requires_grad:
            attn.register_hook(self.save_attn_gradients)

        x = self.matmul2([attn, v])
        x = x.transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def relprop(self, cam, mask, **kwargs): #TODO: make sure relprop that takes this also has a mask
        # shape the same as from the input of forward
        b_, n, c = cam.shape
        cam = self.proj_drop.relprop(cam, **kwargs)

        cam = self.proj.relprop(cam, **kwargs)
        cam = cam.reshape(b_, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        (cam_1, cam_v) = self.matmul2.relprop(cam, **kwargs)

        # Multiply relevance by saved forward attention gradient
        attn_gradients = self.get_attn_gradients()
        cam_1 = cam_1 * attn_gradients
        
        self.save_v_cam(cam_v)
        self.save_attn_cam(cam_1)

        # first reshape, transpose(2, 1) undo matrix multiplication
        if mask is not None:
            nw = mask.shape[0]
            cam_1 = self.softmax.relprop(cam_1, **kwargs)
            cam_1 = cam_1.view(b_ // nw, nw, self.num_heads, n, n)
            (cam_1_res, cam_mask) = self.add.relprop(cam_1, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
            cam_1 = cam_1_res.view(-1, self.num_heads, n, n)
        else:
            cam_1 = self.softmax.relprop(cam_1, **kwargs)
        cam = self.attn_drop.relprop(cam_1, **kwargs)
        (cam_q, cam_k) = self.matmul1.relprop(cam, **kwargs) 

        cam_k = cam_k.transpose(-2, -1)
        cam_qkv = torch.stack([cam_q, cam_k, cam_v], dim=0)
        cam_qkv = cam_qkv.permute(1, 3, 0, 2, 4)
        cam_qkv = cam_qkv.reshape(b_, n, c * 3)
        cam_qkv = self.qkv.relprop(cam_qkv, **kwargs)
        
        return cam_qkv


class SwinTransformerBlock4D(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = LayerNorm,
        use_checkpoint: bool = False,
        labels: dict = {}
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(normalized_shape=dim, labels=labels["norm1"]) # HELPPP labels=labels["norm1"]
        self.attn = WindowAttention4D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            labels=labels["win_attn"] 
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() 
        self.norm2 = norm_layer(normalized_shape=dim, labels=labels["norm"]) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin", labels=labels["mlp"])
        self.dims = None
        self.residual = None
        self.add1 = Add(labels=labels["add1"])
        self.add2 = Add(labels=labels["add2"])
        self.clone1 = Clone(labels=labels["clone1"]) 
        self.clone2 = Clone(labels=labels["clone2"]) 

    def save_residual(self, residual):
        self.residual = residual

    def save_dims(self, dims):
        self.dims = dims

    def forward_part1(self, x, mask_matrix):
        b, d, h, w, t, c = x.shape
        self.window_size, self.shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        # Apply norm1
        x = self.norm1(x)
        
        # Pad the input if necessary
        pad_d0 = pad_h0 = pad_w0 = pad_t0 = 0
        self.pad_d1 = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
        self.pad_h1 = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        self.pad_w1 = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        self.pad_t1 = (self.window_size[3] - t % self.window_size[3]) % self.window_size[3]
        x = F.pad(x, (0, 0, pad_t0, self.pad_t1, pad_w0, self.pad_w1, pad_h0, self.pad_h1, pad_d0, self.pad_d1))  # last tuple first in
        
        # Store padded dimensions
        _, dp, hp, wp, tp, _ = x.shape
        self.dims = [b, dp, hp, wp, tp]
        self.save_dims(self.dims)
        
        # Shift the input if necessary
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], 
                           -self.shift_size[2], -self.shift_size[3]), dims=(1, 2, 3, 4)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        
        # Perform self-attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, *(self.window_size + (c,)))
        shifted_x = window_reverse(attn_windows, self.window_size, self.dims)
        
        # Reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], 
                                   self.shift_size[2], self.shift_size[3]), dims=(1, 2, 3, 4)
            )
        else:
            x = shifted_x
        
        # Remove padding
        if self.pad_d1 > 0 or self.pad_h1 > 0 or self.pad_w1 > 0 or self.pad_t1 > 0:
            x = x[:, :d, :h, :w, :t, :].contiguous()
        
        return x
    
    def relprop_part1(self, cam, mask_matrix, **kwargs):
        # Add back padding if needed
        if self.pad_d1 > 0 or self.pad_h1 > 0 or self.pad_w1 > 0 or self.pad_t1 > 0:
            cam = F.pad(cam, (0, 0, 0, -self.pad_t1, 0, -self.pad_w1, 0, -self.pad_h1, 0, -self.pad_d1))
            
        # Add back cyclic shift if needed
        if any(i > 0 for i in self.shift_size):
            cam = torch.roll(
                cam,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2], -self.shift_size[3]),
                dims=(1, 2, 3, 4)
            )
            
        # Unmerge windows
        cam_windows = window_partition(cam, self.window_size)
        
        '''# Create attention mask if needed
        if mask_matrix is not None:
            print('mask matrix used')
            print(mask_matrix)
            attn_mask = mask_matrix
        else:
            attn_mask = None  '''
        
        # if shift, get attn mask
        if any(i > 0 for i in self.shift_size):
            attn_mask = mask_matrix
        else:
            attn_mask = None  
        
        # Relprop through self-attention
        cam_attn = self.attn.relprop(cam_windows, attn_mask, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
                
        # Merge attention windows
        cam_attn_merged = window_reverse(cam_attn.view(-1, *(self.window_size + (cam_attn.shape[-1],))), self.window_size, self.dims)
        
        # Reverse cyclic shift if needed
        if any(i > 0 for i in self.shift_size):
            cam_shifted = torch.roll(
                cam_attn_merged,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2], self.shift_size[3]),
                dims=(1, 2, 3, 4)
            )
        else:
            cam_shifted = cam_attn_merged
        
        # Remove padding
        cam_unpadded = cam_shifted[:, :self.dims[1], :self.dims[2], :self.dims[3], :self.dims[4], :]
        
        # Remove normalize
        cam_norm1 = self.norm1.relprop(cam_unpadded, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        
        return cam_norm1

    def forward_part2(self, x):
        # Apply norm2
        x = self.norm2(x)

        # Apply MLP
        x = self.mlp(x)

        # Apply stochastic depth
        x = self.drop_path(x)

        return x
    
    def relprop_part2(self, cam, **kwargs):
        cam = self.mlp.relprop(cam, **kwargs)
        cam = self.norm2.relprop(cam, **kwargs)
        return cam

    def forward(self, x, mask_matrix):
        x1, x2 = self.clone1(x, 2)
        if self.use_checkpoint:
            x1 = checkpoint.checkpoint(self.forward_part1, x1, mask_matrix)
        else:
            x1 = self.forward_part1(x1, mask_matrix)
            
        # Add residual connection
        x = self.add1([self.drop_path(x1), x2])
        x1, x2 = self.clone2(x, 2)
        
        if self.use_checkpoint:
            x = self.add2([x1, checkpoint.checkpoint(self.forward_part2, x2)])
        else:
            x = self.add2([x1, self.forward_part2(x2)])

        return x
    

    def relprop(self, cam, mask_matrix, **kwargs):        
        # Undo dropout and add residual connection
        if self.use_checkpoint:
            raise NotImplementedError("Gradient checkpointing is not supported for relevance propagation")
        else:
            (cam1, cam2) = self.add2.relprop(cam, **kwargs)
            cam2 = self.relprop_part2(cam2, **kwargs)
            cam = self.clone2.relprop((cam1, cam2), **kwargs)

        # undo residual
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        
        if self.use_checkpoint:
            raise NotImplementedError("Gradient checkpointing is not supported for relevance propagation")
        else:
            cam1 = self.relprop_part1(cam1, mask_matrix, **kwargs)
            cam = self.clone1.relprop((cam1, cam2), **kwargs)

        return cam
    

class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2, labels: dict = {}
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim

        # Skip dimension reduction on the temporal dimension

        self.reduction = Linear(in_features=8 * dim, out_features=c_multiplier * dim, bias=False, labels=labels["l"])
        self.norm = norm_layer(normalized_shape=8 * dim, labels=labels["norm"]) 
        self.full_data = None
    
    def save_full_data(self, data):
        self.full_data = data

    def forward(self, x):
        x_shape = x.size()
        b, d, h, w, t, c = x_shape
        self.save_full_data(x)
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            -1,
        )

        x = self.norm(x)
        x = self.reduction(x)

        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.reduction.relprop(cam, **kwargs) 
        cam = self.norm.relprop(cam, **kwargs)
        
        b, d, h, w, t, c = cam.shape
        # Split cam into 8 patches
        split_cam = torch.split(cam, cam.shape[-1] // 8, dim=-1)

        # Init empty tensor with initial data size
        undo_merge_cam = torch.zeros_like(self.full_data)

        # Repopulate empty tensor
        for index, (i, j, k) in enumerate(itertools.product(range(2), range(2), range(2))):
            undo_merge_cam[:, i::2, j::2, k::2, :, :] = split_cam[index]

        return undo_merge_cam


MERGING_MODE = {"mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    d, h, w, t = dims
    img_mask = torch.zeros((1, d, h, w, t, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                for t in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
                    img_mask[:, d, h, w, t, :] = cnt
                    cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        labels: dict = {}
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    labels=labels["block"] # HELPPP
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier, labels=labels["down"]
            )
        self.attn_mask = None
        self.matmul = einsum('bhij,bhjd->bhid', labels=labels["ein"])

    def save_attn_mask(self, attn_mask):
        self.attn_mask = attn_mask

    def forward(self, x):
        b, c, d, h, w, t = x.size()
        window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = compute_mask([dp, hp, wp, tp], window_size, shift_size, x.device)
        self.save_attn_mask(attn_mask)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x
    
    def relprop(self, cam, start_layer=0, **kwargs):
        b, c, d, h, w, t = cam.size()

        cam = rearrange(cam, "b c d h w t -> b d h w t c")
        cam = cam.view(b, d, h, w, t, -1)
        
        # downsample relprop
        if self.downsample is not None: 
            cam = self.downsample.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        
        for blk in reversed(self.blocks):
           cam = blk.relprop(cam, self.attn_mask, **kwargs)
        
        cam = rearrange(cam, "b d h w t c -> b c d h w t")
        
        return cam

class BasicLayer_FullAttention(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        labels: dict = {}
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock4D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    labels=labels["block"]
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size), c_multiplier=c_multiplier, labels=labels["down"]
            )
        self.matmul = einsum('bhij,bhjd->bhid', labels=labels["ein"])

    def forward(self, x):
        b, c, d, h, w, t = x.size()

        # window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w t -> b d h w t c")
        # dp = int(np.ceil(d / window_size[0])) * window_size[0]
        # hp = int(np.ceil(h / window_size[1])) * window_size[1]
        # wp = int(np.ceil(w / window_size[2])) * window_size[2]
        # tp = int(np.ceil(t / window_size[3])) * window_size[3]
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, t, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w t c -> b c d h w t")

        return x
    
    def relprop(self, cam, start_layer=0, **kwargs):
        b, c, d, h, w, t = cam.size()
        
        cam = rearrange(cam, "b c d h w t -> b d h w t c")
        cam = cam.view(b, d, h, w, t, -1)
        
        if self.downsample is not None:
            cam = self.downsample.relprop(cam, **kwargs)
            
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, None, **kwargs)
            
        cam = rearrange(cam, "b d h w t c -> b c d h w t")
        
        return cam

class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, patch_dim: tuple, labels: dict
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            patch_num: total number of patches per time frame
            time_num: total number of time frames
        """

        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        d, h, w, t = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1)) # TODO: replace? - should just be a storage doesn't require relprop
        self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))
        self.add1 = Add(labels=labels["add1"])
        self.add2 = Add(labels=labels["add2"])
        # self.clone1 = Clone(labels=labels["clone1"])
        # self.clone2 = Clone(labels=labels["clone2"])
        self.t = None
        
        trunc_normal_(self.pos_embed, std=0.02) # TODO: cannot find this godforsaken fnc on the internet ANYWHEREFML
        
        trunc_normal_(self.time_embed, std=0.02)

    def save_t(self, t):
        self.t = t

    def forward(self, x):
        b, c, d, h, w, t = x.shape
        x = self.add1([x, self.pos_embed])
        x = self.add2([x, self.time_embed[:, :, :, :, :, :t]])
        self.save_t(t)

        return x
    
    def relprop(self, cam, **kwargs):
        # this is done in ViT
        return cam

class SwinTransformer4D(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        img_size: Tuple,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        first_window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = False,
        downsample="mergingv2",
        num_classes=2,
        to_float: bool = False,
        labels: dict = {},
        **kwargs,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


            c_multiplier: multiplier for the feature length after patch merging
        """

        super().__init__()
        
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.first_window_size = first_window_size
        self.patch_size = patch_size
        self.to_float = to_float
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, 
            flatten=False,
            spatial_dims=spatial_dims,
            labels=labels["patch_em"]
        )
        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size
        self.pos_drop = Dropout(p=drop_rate, labels=labels["drop"])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        patch_dim =  ((img_size[0]//patch_size[0]), (img_size[1]//patch_size[1]), (img_size[2]//patch_size[2]), (img_size[3]//patch_size[3]))

        #print img, patch size, patch dim
        print("img_size: ", img_size)
        print("patch_size: ", patch_size)
        print("patch_dim: ", patch_dim)
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_dim, labels=labels["pos_emb"]))
            pos_embed_dim = pos_embed_dim * c_multiplier
            patch_dim = (patch_dim[0]//2, patch_dim[1]//2, patch_dim[2]//2, patch_dim[3])

        # build layer
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
    
        layer = BasicLayer(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.first_window_size,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
            labels=labels["basic_layer"]
        )
        self.layers.append(layer)

        # exclude last layer
        for i_layer in range(1, self.num_layers - 1):
            layer = BasicLayer(
                dim=int(embed_dim * (c_multiplier**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                labels=labels["basic_layer"]
            )
            self.layers.append(layer)

        if not last_layer_full_MSA:
            layer = BasicLayer(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
                labels=labels["basic_layer"]
            )
            self.layers.append(layer)

        else: 
            #################Full MSA for last layer#####################

            self.last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
                self.window_size[3],
            )

            layer = BasicLayer_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.last_window_size,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
                labels=labels["full_att"]
            )
            self.layers.append(layer)

            #############################################################

        self.num_features = int(embed_dim * c_multiplier ** (self.num_layers - 1))
        self.norm = norm_layer(normalized_shape=self.num_features, labels=labels["norm"]) 
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        self.head = Linear(in_features=self.num_features, out_features=1, labels=labels["l"]) if num_classes == 2 else num_classes


    def forward(self, x):
        if self.to_float:
            x = x.float()
        x = self.patch_embed(x)
        x = self.pos_drop(x)  # (b, c, h, w, d, t)
        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x = self.layers[i](x.contiguous())
        return x
    
    def relprop(self, cam, **kwargs):
        for i in reversed(range(self.num_layers)):
            cam = self.layers[i].relprop(cam, **kwargs) 
            cam = self.pos_embeds[i].relprop(cam, **kwargs)
        cam = self.pos_drop.relprop(cam, alpha=kwargs["alpha"], epsilon=kwargs["epsilon"], gamma=kwargs["gamma"])
        cam = self.patch_embed.relprop(cam, **kwargs)
        return cam
