# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch.nn as nn

from monai.networks.layers import get_act_layer
from monai.utils import look_up_option
from .layers_ours import *

SUPPORTED_DROPOUT_MODE = {"vit", "swin"}



class MLPBlock(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """


    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act: tuple | str = "GELU", dropout_mode="vit") -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act: activation type and arguments. Defaults to GELU. Also supports "GEGLU" and others.
            dropout_mode: dropout mode, can be "vit" or "swin".
                "vit" mode uses two dropout instances as implemented in
                https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py#L87
                "swin" corresponds to one instance as implemented in
                https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_mlp.py#L23


        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = Linear(hidden_size, mlp_dim) if act != "GEGLU" else Linear(hidden_size, mlp_dim * 2)
        self.linear2 = Linear(mlp_dim, hidden_size)
        # TODO currently no definition of get_act_layer in SwifT-LRP
        self.fn = get_act_layer(act) #  change this to GELU if we're sure
        self.drop1 = Dropout(dropout_rate)
        dropout_opt = look_up_option(dropout_mode, SUPPORTED_DROPOUT_MODE)
        if dropout_opt == "vit":
            self.drop2 = Dropout(dropout_rate)
        elif dropout_opt == "swin":
            self.drop2 = self.drop1
        else:
            raise ValueError(f"dropout_mode should be one of {SUPPORTED_DROPOUT_MODE}")




    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

    def relprop(self, cam, **kwargs):
        # TODO: start thinking about how we'll structure these relprop functions
        # to accept our invasive genetic engineering
        cam = self.drop2.relprop(cam, **kwargs)
        cam = self.linear2.relprop(cam, **kwargs)
        # FIXME- passes should be fine for dropout? yea so we can probably leave this out
        cam = self.drop1.relprop(cam, **kwargs) # tutorial doesn't have this -> both are passes
        # TODO: i think this is GELU 
        cam = self.fn.relprop(cam, **kwargs)
        cam = self.linear1.relprop(cam, **kwargs)
        return cam
