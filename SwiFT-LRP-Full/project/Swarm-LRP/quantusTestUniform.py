import pathlib
import numpy as np
import multiprocessing

import math

# import quantus
import torch
torch.cuda.empty_cache()
# print(torch.cuda.memory_summary(device=None, abbreviated=False))
import gc
gc.collect()

import torch.multiprocessing as tmp
import torchvision
# from captum.attr import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# sns.set()

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
# from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# from module import LitClassifier
# import neptune.new as neptune
from module.utils.data_module import fMRIDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier

# import LRP class
from module.Explanation_generator import LRP
from module.utils.visualization import visualize

import sys
from particleSwarm import pso, pso_layers
# adding Folder_2 to the system path
# from quantus.metrics.robustness.max_sensitivity_3d import MaxSensitivity
import quantus


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

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

labels = {"patch_em": {"l": "gamma"}, "drop": "gamma", "pos_emb": {"add1": "gamma", "add2": "gamma"},
       "basic_layer": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
       "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"}, "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"},
       "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"},
       "full_att": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
       "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"},
       "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"}, "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"}, "norm": "gamma", "avg_pool": "gamma", "l": "gamma",
       "clf_mlp": {"l": "gamma", "sm": "gamma", "do": "gamma"}}

'''labels = {"patch_em": {"l": "epsilon"}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"},
       "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "gamma", "do": "alpha", "l2": "epsilon", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
       "norm": "alpha", "mlp": {"l1": "epsilon", "l2": "gamma", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"},
       "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"},
       "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
       "norm": "alpha", "mlp": {"l1": "epsilon", "l2": "gamma", "gelu": "alpha", "do": "alpha"},
       "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "gamma",
       "clf_mlp": {"l": "zero", "sm": "alpha", "do": "alpha"}}'''

# layer opt
'''labels = {"patch_em": {"l": 'gamma'}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
        "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": 'gamma', "do": "alpha", "l2": 'zero', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
        "norm": "alpha", "mlp": {"l1": 'epsilon', "l2": 'alpha', "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, 
        "down": {"l": 'alpha', "norm": "alpha"}, "ein": "alpha"}, 
        "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": 'alpha', "do": "alpha", "l2": 'zero', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
        "norm": "alpha", "mlp": {"l1": 'epsilon', "l2": 'zero', "gelu": "alpha", "do": "alpha"},
        "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": 'zero', "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": 'gamma',
        "clf_mlp": {"l": 'gamma', "sm": "alpha", "do": "alpha"}}'''

args.labels = labels

# ------------ data -------------
print("ARGS: ", args)
data_module = Dataset(**vars(args))

model = Classifier(data_module = data_module, **vars(args)) 
model.to(device)
lrp_instance = LRP(model)
x_batch, y_batch = None, None
lrp_result = None
input_data_single = None

# Fetch data loader
data_loader = iter(data_module.test_dataloader())
for single_data in data_loader:
    input_data = single_data["fmri_sequence"]
    input_labels = single_data["schizo"]
    # print("input: ", input_labels)

    # Select the first item in the batch for LRP analysis
    # Assuming that your model and the data_loader support processing single items directly
    # input_data_single = input_data.to(device)  # Ensure input has batch dimension if needed input_data[0].unsqueeze(0).to(device)
    input_data_single = input_data[0].unsqueeze(0).to(device)
    # Determine the predicted class or the index for LRP
    # predicted_index = output.argmax(dim=1).item()  # Get the index of the max logit which is the predicted class

    x_batch = input_data_single
    # y_batch = input_labels[0].to(device) # [tensor]
    y_batch = input_labels[0][0].to(device) # [tensor]
    print("label:", y_batch)

    # print(x_batch.shape, y_batch)
    # Execute LRP
    #lrp_result = lrp_instance.generate_LRP(input_data_single, alpha=1.5, epsilon=1E-7, gamma=0.75, device=device)
    lrp_result = lrp_instance.generate_LRP(input_data_single, alpha=1.08577486e+00, epsilon=4.24792029e-07, gamma=0.75, device=device)
    # print("LRP result:", lrp_result.shape)
    visualize(lrp_result)
    break # should only do one data point (could edit this to do them all)
"""
# generate translrp explanation
# a_batch_saliency = quantus.normalise_func.normalise_by_negative(Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())

# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.

# print(torch.sum(input_data_single))
# print(torch.sum(lrp_result))

lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
x_batch, y_batch = x_batch.cpu().numpy(), y_batch.unsqueeze(0).cpu().numpy()


# Quick assert.
# assert [isinstance(obj, np.ndarray) for obj in [x_batch, y_batch, a_batch_saliency, a_batch_intgrad]]

def generate_lrp_results(lrp_instance, x, y, input):
    lrp_result = lrp_instance.generate_LRP(input, alpha=x[0], epsilon=x[1], gamma=x[2])
    lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
    
    # normalize lrp result
    lrp_result = lrp_result / np.sum(lrp_result)

    results = evaluate_metrics(x, y, input, lrp_result)
    return results

def evaluate_metric(metric_init, input, x, y, lrp_result):
    return metric_init(model=model, 
                        x_batch=input,
                        y_batch=y,
                        a_batch=lrp_result,
                        device=device,
                        explain_func=quantus.explain,
                        explain_func_kwargs={"method": "TransLRP", "softmax": False, "alpha": x[0], "epsilon": x[1], "gamma": x[2]})[0]
    
def evaluate_metrics(x, y, input, lrp_result):
    faith_corr = quantus.FaithfulnessCorrelation(
        similarity_func=quantus.similarity_func.correlation_pearson,
        nr_runs=60,
        subset_size= 2240,
        abs=True,
        normalise=True,
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        perturb_baseline="black"
    )
    
    infidelity = quantus.Infidelity(    
        perturb_baseline="uniform",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        n_perturb_samples=5,
        perturb_patch_sizes=[32], 
        display_progressbar=True,)

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

    results = [evaluate_metric(metric, input, x, y, lrp_result) for metric in [faith_corr, eff_complex, avg_sens]]
    results[1] /= np.prod(input.numel())
    results[0] = np.abs(results[0])

    print("faithfulness: ", results[0], "complexity: ", results[1], "sensitivity: ", results[2])
    return np.array(results)

# Define the optimizer function
def optimizer(x, y, input, prev):
    # compute a_batch with new params
    lrp_result = lrp_instance.generate_LRP(input, alpha=x[0], epsilon=x[1], gamma=x[2])
    lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
    
    # normalize lrp result
    lrp_result = lrp_result / np.sum(lrp_result)

    results = evaluate_metrics(x, y, input, lrp_result)
    normalized_results = ((results - prev) / prev) 
    print(normalized_results)
    # need an optimizer function sensitivity, complexity lower, faithfulness higher
    return -normalized_results[0] + normalized_results[1] + normalized_results[2]

def convert_layers(particles):
    # change the layers being optimized on
    # Dictionary to map floored integers to corresponding strings
    layer_map = {
        0: 'zero',
        1: 'epsilon',
        2: 'alpha',
        3: 'gamma'
    }
    
    particles = [layer_map[value.astype(int)] for value in particles]
    print("\n\nChosen layers:", particles)
    l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12 = particles
    
    return {"patch_em": {"l": l0}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
        "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": l1, "do": "alpha", "l2": l2, "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
        "norm": "alpha", "mlp": {"l1": l3, "l2": l4, "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, 
        "down": {"l": l5, "norm": "alpha"}, "ein": "alpha"}, 
        "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": l6, "do": "alpha", "l2": l7, "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
        "norm": "alpha", "mlp": {"l1": l8, "l2": l9, "gelu": "alpha", "do": "alpha"},
        "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": l10, "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": l11,
        "clf_mlp": {"l": l12, "sm": "alpha", "do": "alpha"}}

def optimizer_layer(x, y, input, prev):
    args.labels = convert_layers(x)

    model = Classifier(data_module = data_module, **vars(args)) 
    model.to(device)
    lrp_instance = LRP(model)
    # compute a_batch with new params
    lrp_result = lrp_instance.generate_LRP(input, alpha=1.5, epsilon=0.0000001, gamma=0.75)
    lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
    
    # normalize lrp result
    lrp_result = lrp_result / np.sum(lrp_result)

    results = evaluate_metrics(x, y, input, lrp_result)
    normalized_results = ((results - prev) / prev) 
    # need an optimizer function sensitivity, complexity lower, faithfulness higher
    return -normalized_results[0] + normalized_results[1] + normalized_results[2]

def test(x, y, input):
    # compute a_batch with new params
    lrp_result = lrp_instance.generate_LRP(input, alpha=x[0], epsilon=x[1], gamma=x[2])
    lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
    
    infidelity = quantus.Infidelity(    
        perturb_baseline="uniform",
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        n_perturb_samples=5,
        perturb_patch_sizes=[32], 
        display_progressbar=True,)

    results = evaluate_metric(infidelity, input, x, y, lrp_result) 
    # results[1] /= np.prod(input.numel())
    print("infidelity: ", results)

if __name__ == "__main__":
    # Normalize (sum to 1)
    lrp_result = lrp_result / np.sum(lrp_result)
    input_data_single = input_data_single / torch.sum(input_data_single)
    
    dim = 3
    
    # saved overrides:
    value_after_3 = np.array([[ 1.00000000e+0,  7.99812145e-08,  1.07015182e+00], [ 1.02272580e+00,  1.22734247e-07,  1.06479909e+00], 
                              [ 1.31186443e+00,  1.53385398e-07,  9.33345029e-01], [ 1.00000000e+00,  3.17406140e-07, -4.72816742e-01], 
                              [ 1.14843284e+00,  1.18295002e-07,  9.97399698e-01]])
    
    layer_after_1 = np.array([[3.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 2.0], [2.0, 3.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 3.0], 
                              [3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0], [3.0, 3.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0, 2.0], 
                              [3.0, 3.0, 0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0]])
    fitness_after_1 = np.array([-0.03513471,  0.72015421, -1.89053422,  1.28824263, -1.06109625])
    
    # layer_after_3 = np.array([[2.0, 1.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 3.0], [2.0, 2.0, 3.0, 3.0, 2.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 2.0], [1.0, 1.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 2.0], [1.0, 2.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 2.0]])
    # fitness_after_3 = np.array([-1.26329751, -0.97091406, -9.38477183, -2.32177917, -0.58000874])
    
    #print(value_after_3)
    
    # evaluate_metrics([1.90906828, 0.71202453, 0.19965881], y_batch, input_data_single, lrp_result)
    # test([1, 1, 1], y_batch, input_data_single)
    print("****** BASELINE ******")
    baselines = evaluate_metrics([2, 0, 0.75], y_batch, input_data_single, lrp_result)
    # baselines = (1, 1, 1)
 
    print("******* PSO ******")
    # Run the PSO algorithm 
    
    # No override (value)
    solution, fitness = pso(optimizer, dim=dim, num_particles=5, 
                            max_iter=3, input=input_data_single, y=y_batch, baselines=baselines)
    
    # Override (value)
    # solution, fitness = pso(optimizer, dim=dim, num_particles=5, 
    #                         max_iter=2, input=input_data_single, y=y_batch, baselines=baselines, override=True, override_particles=value_after_3)
    
    # No override (layers)
    # solution, fitness = pso_layers(optimizer_layer, dim=13, num_particles=5, 
    #                     max_iter=1, input=input_data_single, y=y_batch, baselines=baselines)
    
    # Override (layers)
    # solution, fitness = pso_layers(optimizer_layer, dim=13, num_particles=5, 
    #                     max_iter=2, input=input_data_single, y=y_batch, baselines=baselines, override=True, override_particles=layer_after_1, override_fitness=fitness_after_1)

    # # Print the solution and fitness value
    print('Solution:', solution)
    print('Fitness:', fitness)"""