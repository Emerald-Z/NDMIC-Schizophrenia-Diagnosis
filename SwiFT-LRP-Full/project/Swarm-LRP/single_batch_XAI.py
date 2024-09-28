from captum.attr import IntegratedGradients

import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl

# from module import LitClassifier
# import neptune.new as neptune
from module.utils.data_module import fMRIDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier

from module.utils.visualization import visualize
from utils import generate_feature_mask
from tqdm import tqdm
import sys

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("QUANTUS DEVICE:", device)

# ------------ args -------------
parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
parser.add_argument("--dataset_name", type=str, choices=["S1200", "ABCD", "UKB", "Dummy", "COBRE"], default="COBRE")
parser.add_argument("--downstream_task", type=str, default="schizo", help="downstream task")
parser.add_argument("--downstream_task_type", type=str, default="default", help="select either classification or regression according to your downstream task")
parser.add_argument("--classifier_module", default="default", type=str, help="A name of lightning classifier module (outdated argument)")
parser.add_argument("--loggername", default="default", type=str, help="A name of logger")
parser.add_argument("--project_name", default="default", type=str, help="A name of project (Neptune)")
parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
parser.add_argument("--test_only", action='store_true', help="specify when you want to test the checkpoints (model weights)")
parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
# LRP arg
parser.add_argument("--lrp_test", action='store_true', help="Perform LRP test on a single piece of data")
parser.add_argument("--explanation_method", default="lrp", type=str, choices=["TransLRP", "IntegratedGradients", "Saliency", "Lime"], help="choose explanation method to run")

temp_args, _ = parser.parse_known_args()

# Set classifier
Classifier = LitClassifier

# Set dataset
Dataset = fMRIDataModule
print("QUANTUS DATASET CREATED")

# add two additional arguments
parser = Classifier.add_model_specific_args(parser)
print("ADDED CLASSIFIER ARGS")
parser = Dataset.add_data_specific_args(parser)
print("ADDED DATA SPECIFIC ARGS")

_, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
print("PARSED KNOWN ARGS")
parser = pl.Trainer.add_argparse_args(parser)
print("ARGPARSE ARGS ADDED TO pl.Trainer")
args = parser.parse_args()
print("PARSED ARGS")

pl.seed_everything(args.seed)
torch.use_deterministic_algorithms(True) # FIXME: needed?

#override parameters
max_epochs = args.max_epochs
num_nodes = args.num_nodes
devices = args.devices
print("DEVICES:", devices)

print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

project_name = args.project_name
image_path = args.image_path

if temp_args.resume_ckpt_path is not None:
    # resume previous experiment
    from module.utils.neptune_utils import get_prev_args
    args = get_prev_args(args.resume_ckpt_path, args)
    exp_id = args.id
    # override max_epochs if you hope to prolong the training
    args.project_name = project_name
    args.max_epochs = max_epochs
    args.num_nodes = num_nodes
    args.devices = devices
    args.image_path = image_path       
else:
    exp_id = None

setattr(args, "default_root_dir", f"output/{args.project_name}")

# ------------ labels -------------

labels = {"patch_em": {"l": "gamma"}, "drop": "alpha", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
"basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
"norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, 
"full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "epsilon", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
"norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"},
"add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "alpha",
"clf_mlp": {"l": "alpha", "sm": "alpha", "do": "alpha"}}

args.labels = labels

# ------------ data -------------
print("ARGS: ", args)
data_module = Dataset(**vars(args))
method = args.explanation_method

model = Classifier(data_module = data_module, **vars(args)) 
print(device)
model.to(device)
x_batch, y_batch = None, None
lrp_result = None
input_data_single = None

# Fetch data loader
data_loader = iter(data_module.test_dataloader())
for single_data in data_loader:
    print(single_data)
    input_data = single_data["fmri_sequence"]
    input_labels = single_data["schizo"]

    input_data_single = input_data[0].unsqueeze(0) # Ensure input has batch dimension if needed

    x_batch = input_data_single.to(device) 
    y_batch = input_labels[0][0].to(device)
    print(y_batch)
    print("x batch shape:", x_batch.shape)
    print("y batch:", y_batch)

    break # should only do one data point (could edit this to do them all)

input_data_single = input_data_single / torch.sum(input_data_single)

if method == "IntegratedGradients":
    integrated_gradients  = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    kwargs = {
        "nt_samples": 5,
        "nt_samples_batch_size": 5,
        "nt_type": "smoothgrad_sq", # 1
        #"stdevs": 0.05,
        "internal_batch_size": 5,
    }

    result = noise_tunnel.attribute(input_data_single.to(device), baselines=input_data_single[0,0,0,0,0,0].item(),target=None,**kwargs)
elif method == "Lime":
    zero_target = torch.from_numpy(np.array([0])).to(device)
    print("shapes: ", x_batch.device, zero_target.device)
    label_idx = y_batch
    result = lr_lime.attribute(
        x_batch,
        target=zero_target,
        feature_mask=generate_feature_mask().to(device),
        n_samples=40,
        perturbations_per_eval=16,
        show_progress=True
    ).squeeze(0)

print("result calculated: ", result.shape)
visualize(result)
"""print("result calculated")
result = result.cpu().numpy()
x_batch, y_batch = x_batch, y_batch.unsqueeze(0).cpu().numpy()

# faith_corr = quantus.FaithfulnessCorrelation(
#     nr_runs = 1,
#     perturb_func=quantus.perturb_func.uniform_noise,
#     similarity_func=quantus.similarity_func.difference,
#     disable_warnings=True,
#     normalise=True,
#     abs=True)

faithfulness_corr = quantus.FaithfulnessCorrelation(
    similarity_func=quantus.similarity_func.correlation_pearson,
    nr_runs=60,
    subset_size= 2240,
    abs=True,
    normalise=True,
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    perturb_baseline="black"
)

eff_complex = quantus.EffectiveComplexity(
    eps=1e-7,
    disable_warnings=True,
    normalise=True,
    abs=True)

avg_sens = quantus.AvgSensitivity(
    nr_samples=10,
    lower_bound=0.2,
    norm_numerator=quantus.norm_func.fro_norm,
    norm_denominator=quantus.norm_func.fro_norm,
    perturb_func=quantus.perturb_func.uniform_noise,
    similarity_func=quantus.similarity_func.difference,
    disable_warnings=True,
    normalise=True,
    abs=True)

model.eval()
result = result / np.sum(result)
print(result)
x_batch = x_batch / torch.sum(x_batch)
fc = faithfulness_corr(model=model, 
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=result,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={"method": "IntegratedGradients", "softmax": False})
print("fc done")
ec = eff_complex(model=model, 
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=result,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={"method": "IntegratedGradients", "softmax": False})
print('ec done')

aso = avg_sens(model=model, 
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=result,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={"method": "IntegratedGradients"})
print(ec)
print(np.abs(fc), ec[0] / (96*96*96*20), aso)
#print(ec)
# print(aso)"""