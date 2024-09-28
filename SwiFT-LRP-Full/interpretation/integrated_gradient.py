import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
 
import os
from tqdm import tqdm
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import sys

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, '{your_path_here}/SwiFT-LRP-Full')

# from project.module.models.swin4d_transformer_ver7 import SwinTransformer4D
from project.module.pl_classifier import LitClassifier
from project.module.utils.data_module import fMRIDataModule

from pathlib import Path

save_dir = 'interpretation/ig_results' # write path to save_dir
# jobid = # write project number
# neptune_project_id = # write project id. ex)user_id/project_name
# for i in Path(f'SwiFT/output/{neptune_project_id}/RSTOT-{jobid}/').glob('checkpt*'):
#         ckpt_path = i

ckpt_path = 'output/default/last_LRP_TEST.ckpt'
ckpt = torch.load(ckpt_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt['hyper_parameters']['image_path'] = # write path to MNI_to_TRs folder 
ckpt['hyper_parameters']['default_root_dir'] = # write path to use default_root_dir
ckpt['hyper_parameters']['shuffle_time_sequence'] = False
ckpt['hyper_parameters']['time_as_channel'] = False
ckpt['hyper_parameters']['eval_batch_size'] = 1
ckpt['hyper_parameters']['stride_within_seq'] = 1
ckpt['hyper_parameters']['stride_between_seq'] = 1
ckpt['hyper_parameters']['input_type'] = "rest"
ckpt['hyper_parameters']['label_scaling_method'] = "standardization"

args = ckpt['hyper_parameters']


data_module = fMRIDataModule(**args)
data_module.setup()
data_module.prepare_data()
test_loader = data_module.test_dataloader()


labels = {"patch_em": {"l": "alpha"}, "drop": "alpha", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
"basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
"norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, 
"full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
"norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"},
"add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "alpha",
"clf_mlp": {"l": "alpha", "sm": "alpha", "do": "alpha"}}

cuda = 0
model = LitClassifier(data_module=data_module, labels=labels, **args)
model.cuda(cuda) if torch.cuda.is_available() else model
model.load_state_dict(ckpt['state_dict'])

integrated_gradients  = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)

kwargs = {
    "nt_samples": 5,
    "nt_samples_batch_size": 5,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 5,
}


model.eval()
for idx, data in enumerate(tqdm(test_loader),0):
    subj_name = data['subject_name'][0]
    dataset_name = ckpt['hyper_parameters']['dataset_name']
    tr = data['TR'].item()
    input_ts = data['fmri_sequence'].float().cuda(cuda)
    label = data['schizo'][0].float().cuda(cuda) # [0] added
    
    pred = model.forward(input_ts)
    pred_prob = torch.sigmoid(pred)
    pred_int = (pred_prob>0.5).int().item()
    
    target = data['schizo'][0]
    target_int = target.int().item()
    
    print("Probabilities Prediction, Sigmoid probability", pred, pred_prob, target_int)
    
    #only choose corrected samples
    
    if pred_int == target_int:
        if target_int == 0:
            if pred_prob <= 0.25:
                file_dir = os.path.join(save_dir, f'{dataset_name}_target0')
                os.makedirs(file_dir,exist_ok=True)
                if tr % 100 == 0:
                    file_path = os.path.join(file_dir, f"{subj_name}_{tr}.pt") 
                    if not os.path.exists(file_path):
                        result = noise_tunnel.attribute(input_ts,baselines=input_ts[0,0,0,0,0,0].item(),target=None,**kwargs)
                        result = result.squeeze().cpu()
                        torch.save(result, file_path)
                        print(f'saving {subj_name}_{tr}.pt')
        elif target_int == 1:
            if pred_prob >= 0.75:
                file_dir = os.path.join(save_dir, f"{dataset_name}_target1")
                os.makedirs(file_dir,exist_ok=True)
                if tr % 100 == 0:
                    file_path = os.path.join(file_dir, f"{subj_name}_{tr}.pt") 
                    if not os.path.exists(file_path):
                        result = noise_tunnel.attribute(input_ts,baselines=input_ts[0,0,0,0,0,0].item(),target=None,**kwargs)
                        result = result.squeeze().cpu()
                        torch.save(result, file_path)
                        print(f'saving {subj_name}_{tr}.pt')
    # break
