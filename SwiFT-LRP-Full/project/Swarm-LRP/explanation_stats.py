import numpy as np
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pytorch_lightning as pl
import sys
sys.path.insert(0, '/raid/neuro/Schizophrenia_Project/NDMIC-Schizophrenia-Diagnosis/SwiFT-LRP-Full')

from project.module.utils.data_module import fMRIDataModule
from project.module.pl_classifier import LitClassifier

# import LRP class
from project.module.Explanation_generator import LRP

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from utils import generate_feature_mask
from quantusEval import QuantusEval
import quantus
from captum._utils.models.linear_model import SkLearnLinearRegression
from visualization import visualize

import time
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

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
parser.add_argument("--explanation_method", type=str, choices=["TransLRP", "IntegratedGradients", "Lime", "GradCam", "Saliency"], help="choose explanation method to run")

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
method = args.explanation_method
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

# labels = {"patch_em": {"l": "gamma"}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"},
#        "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "gamma", "do": "alpha", "l2": "gamma", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
#        "norm": "alpha", "mlp": {"l1": "gamma", "l2": "alpha", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"},
#        "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"},
#        "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "zero", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
#        "norm": "alpha", "mlp": {"l1": "epsilon", "l2": "zero", "gelu": "alpha", "do": "alpha"},
#        "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "zero", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "gamma",
#        "clf_mlp": {"l": "gamma", "sm": "alpha", "do": "alpha"}}

# labels = {"patch_em": {"l": 'gamma'}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
#         "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": 'gamma', "do": "alpha", "l2": 'zero', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
#         "norm": "alpha", "mlp": {"l1": 'epsilon', "l2": 'alpha', "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, 
#         "down": {"l": 'alpha', "norm": "alpha"}, "ein": "alpha"}, 
#         "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": 'alpha', "do": "alpha", "l2": 'zero', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
#         "norm": "alpha", "mlp": {"l1": 'epsilon', "l2": 'zero', "gelu": "alpha", "do": "alpha"},
#         "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": 'zero', "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": 'gamma',
#         "clf_mlp": {"l": 'gamma', "sm": "alpha", "do": "alpha"}}
       
# uniform gamma
# labels = {"patch_em": {"l": "gamma"}, "drop": "gamma", "pos_emb": {"add1": "gamma", "add2": "gamma"},
#        "basic_layer": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
#        "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"}, "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"},
#        "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"},
#        "full_att": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
#        "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"},
#        "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"}, "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"}, "norm": "gamma", "avg_pool": "gamma", "l": "gamma",
#        "clf_mlp": {"l": "gamma", "sm": "gamma", "do": "gamma"}}

# uniform epsilon
labels = {"patch_em": {"l": "epsilon"}, "drop": "epsilon", "pos_emb": {"add1": "epsilon", "add2": "epsilon"},
       "basic_layer": {"block": {"norm1": "epsilon", "win_attn": {"l1": "epsilon", "do": "epsilon", "l2": "epsilon", "sm": "epsilon", "add": "epsilon", "ein1": "epsilon", "ein2": "epsilon"},
       "norm": "epsilon", "mlp": {"l1": "epsilon", "l2": "epsilon", "gelu": "epsilon", "do": "epsilon"}, "add1": "epsilon", "add2": "epsilon", "clone1": "epsilon", "clone2": "epsilon"},
       "down": {"l": "epsilon", "norm": "epsilon"}, "ein": "epsilon"},
       "full_att": {"block": {"norm1": "epsilon", "win_attn": {"l1": "epsilon", "do": "epsilon", "l2": "epsilon", "sm": "epsilon", "add": "epsilon", "ein1": "epsilon", "ein2": "epsilon"},
       "norm": "epsilon", "mlp": {"l1": "epsilon", "l2": "epsilon", "gelu": "epsilon", "do": "epsilon"},
       "add1": "epsilon", "add2": "epsilon", "clone1": "epsilon", "clone2": "epsilon"}, "down": {"l": "epsilon", "norm": "epsilon"}, "ein": "epsilon"}, "norm": "epsilon", "avg_pool": "epsilon", "l": "epsilon",
       "clf_mlp": {"l": "epsilon", "sm": "epsilon", "do": "epsilon"}}

# labels = {"patch_em": {"l": 'gamma'}, 
#             "drop": "gamma", 
#             "pos_emb": {"add1": "gamma", "add2": "gamma"}, 
#         "basic_layer": {
#             "block": {
#                 "norm1": "gamma", "win_attn": {
#                     "l1": 'gamma', "do": "gamma", "l2": 'gamma', "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"}, 
#                 "norm": "epsilon", "mlp": {"l1": 'gamma', "l2": 'epsilon', "gelu": "epsilon", "do": "epsilon"}, "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"}, 
#             "down": {
#                 "l": 'gamma', "norm": "gamma"}, 
#             "ein": "gamma"}, 
#         "full_att": {
#             "block": {
#                 "norm1": "epsilon", "win_attn": {
#                     "l1": 'epsilon', "do": "epsilon", "l2": 'epsilon', "sm": "epsilon", "add": "epsilon", "ein1": "epsilon", "ein2": "epsilon"}, 
#             "norm": "gamma", "mlp": {"l1": 'epsilon', "l2": 'epsilon', "gelu": "epsilon", "do": "epsilon"},
#             "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"}, 
#             "down": {"l": 'zero', "norm": "zero"}, "ein": "zero"}, 
#         "norm": "zero", "avg_pool": "zero", "l": 'zero', 
#         "clf_mlp": {
#             "l": 'zero', "sm": "zero", "do": "zero"}}

# labels = {"patch_em": {"l": 'alpha'}, "drop": "alpha", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
#         "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": 'alpha', "do": "alpha", "l2": 'alpha', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
#         "norm": "alpha", "mlp": {"l1": 'alpha', "l2": 'alpha', "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, 
#         "down": {"l": 'alpha', "norm": "alpha"}, "ein": "alpha"}, 
#         "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": 'alpha', "do": "alpha", "l2": 'alpha', "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
#         "norm": "alpha", "mlp": {"l1": 'alpha', "l2": 'alpha', "gelu": "alpha", "do": "alpha"},
#         "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": 'alpha', "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": 'gamma',
#         "clf_mlp": {"l": 'gamma', "sm": "alpha", "do": "alpha"}}

args.labels = labels
# ------------ data -------------
print("ARGS: ", args)
data_module = Dataset(**vars(args))
print(device)
model = Classifier(data_module = data_module, **vars(args)) 
model.to(device)
lrp_instance = LRP(model)
x_batch, y_batch = None, None
lrp_result = None
input_data_single = None

LRP_PARAMS = [1.86, 0.0000000972, 0.88]
evaluator = QuantusEval(model)

# Fetch data loader
data_loader = iter(data_module.test_dataloader())
dataset = data_module.test_dataloader().dataset
num_samples = len(dataset)

metric_values = np.zeros((23, 3))
metric_labels = np.zeros((23, 2))
subject_idx = 0
subjects = []
integrated_gradients  = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)

kwargs = {
    "nt_samples": 5,
    "nt_samples_batch_size": 5,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 5,
}

def create_feature_mask():
    height, width, depth, channels = 96, 96, 96, 20
    num_blocks = 10000  # Number of unique values to fill the tensor

    # Create an empty tensor with the desired shape
    tensor = torch.zeros((height, width, depth, channels), dtype=torch.int32)

    # Calculate the number of blocks per dimension
    block_size = int((height * width * depth * channels) / num_blocks)

    # Initialize the value for each block
    value = 1

    # Fill the tensor with block values
    for idx in range(num_blocks):
        # Define the starting and ending indices for the block
        start_idx = idx * block_size
        end_idx = start_idx + block_size
        
        # Flatten the tensor and assign the block values
        flattened_tensor = tensor.view(-1)
        flattened_tensor[start_idx:end_idx] = value
        
        # Increment value for the next block
        value += 1
    return tensor

for (data_idx, single_data) in enumerate(data_loader):
    input_data = single_data["fmri_sequence"]
    input_labels = single_data["schizo"]

    for i in range(input_labels[0].shape[0]):
        # Select the first item in the batch for LRP analysis
        # Assuming that your model and the data_loader support processing single items directly
        # input_data_single = input_data.to(device)  # Ensure input has batch dimension if needed input_data[0].unsqueeze(0).to(device)
        curr_subj = single_data["subject_name"][i]
        if curr_subj not in subjects:
            print(curr_subj)
            input_data_single = input_data[i].unsqueeze(0).to(device)
            # Determine the predicted class or the index for LRP
            # predicted_index = output.argmax(dim=1).item()  # Get the index of the max logit which is the predicted class
            
            output = model.forward(input_data_single)
            index = 1 if output.cpu().data.numpy()[0][0] > 0 else 0
            x_batch = input_data_single
            # y_batch = input_labels[0].to(device) # [tensor]
            y_batch = input_labels[0][i].to(device) # [tensor]
            metric_labels[subject_idx, 0] = y_batch
            metric_labels[subject_idx, 1] = index
            print(y_batch, index)

            if method == "TransLRP":
                # Execute LRP
                lrp_result = lrp_instance.generate_LRP(input_data_single, alpha=LRP_PARAMS[0], epsilon=LRP_PARAMS[1], gamma=LRP_PARAMS[2], device=device) # there's a way to pass them all in **
                lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
            elif method == "IntegratedGradients":
                # IG
                lrp_result = noise_tunnel.attribute(input_data_single, baselines=input_data_single[0,0,0,0,0,0].item(),target=None,**kwargs)
                lrp_result = lrp_result.cpu().detach().numpy()
            elif method == "Lime":
                    lrp_result = quantus.explain(model, x_batch,  np.array([0]), method="Lime", feature_mask=generate_feature_mask().to(device),
                                                 interpretable_model=SkLearnLinearRegression())
                    y_batch = torch.from_numpy(np.array([0]))   
            elif method == "Saliency":
                lrp_result = quantus.explain(model, x_batch,  np.array([0]), method="Saliency")
            else:
                raise ValueError(
                        f"'Method {method} not recognized"
                    )

            x_batch, y_batch = x_batch.cpu().numpy(), y_batch.unsqueeze(0).cpu().numpy()

            # normalize lrp result
            lrp_result = lrp_result / np.sum(lrp_result)
            input_data_single = input_data_single / torch.sum(input_data_single)

            metric_values[subject_idx, :] =  evaluator.evaluate_metrics(LRP_PARAMS, y_batch, input_data_single, lrp_result, method)
            print(metric_values[subject_idx, :])
            subject_idx += 1
            subjects.append(curr_subj)

print('It took', time.time()-start, 'seconds.')

print("****** TOTAL ******")
print("mean : ", np.mean(metric_values, 0))
print("standard dev", np.std(metric_values, 0))

print("****** PER LABEL ******")
filter_mask = (metric_labels[:, 0] == 0)
filtered = metric_values[filter_mask, :]
print("mean 0: ", np.mean(filtered, 0))
print("standard dev 0", np.std(filtered, 0))
filter_mask = (metric_labels[:, 0] == 1)
filtered = metric_values[filter_mask, :]
print("mean 1: ", np.mean(filtered, 0))
print("standard dev 1", np.std(filtered, 0))

print("****** CORRECT LABELS ******")
filter_mask = (metric_labels[:, 0] == 0) & (metric_labels[:, 1] == 0)
filtered = metric_values[filter_mask, :]
print("mean 0: ", np.mean(filtered, 0))
print("standard dev 0", np.std(filtered, 0))

filter_mask = (metric_labels[:, 0] == 1) & (metric_labels[:, 1] == 1)
print(filter_mask)

filtered = metric_values[filter_mask, :]
print("mean 1: ", np.mean(filtered, 0))
print("standard dev 1", np.std(filtered, 0))

print("****** INCORRECT ******")
filter_mask = (metric_labels[:, 0] == 0) & (metric_labels[:, 1] == 1)
print(filter_mask)
filtered = metric_values[filter_mask, :]
print("mean 0: ", np.mean(filtered, 0))
print("standard dev 0", np.std(filtered, 0))
filter_mask = (metric_labels[:, 0] == 1) & (metric_labels[:, 1] == 0)
filtered = metric_values[filter_mask, :]
print("mean 1: ", np.mean(filtered, 0))
print("standard dev 1", np.std(filtered, 0))


print("****** TOTAL CORRECT LABELS ******")
filter_mask = (metric_labels[:, 0] == metric_labels[:, 1])
filtered = metric_values[filter_mask, :]
print("mean: ", np.mean(filtered, 0))
print("standard dev", np.std(filtered, 0))

print("****** TOTAL INCORRECT LABELS ******")
filter_mask = (metric_labels[:, 0] != metric_labels[:, 1])
filtered = metric_values[filter_mask, :]
print("mean: ", np.mean(filtered, 0))
print("standard dev", np.std(filtered, 0))

print("****** TOTAL 0 LABELS ******")
filter_mask = np.where(metric_labels[:, 0] == 0)[0]
filtered = metric_values[filter_mask, :]
print("mean: ", np.mean(filtered, 0))
print("standard dev", np.std(filtered, 0))

print("****** TOTAL 1 LABELS ******")
filter_mask = np.where(metric_labels[:, 0] == 1)[0]
filtered = metric_values[filter_mask, :]
print("mean: ", np.mean(filtered, 0))
print("standard dev", np.std(filtered, 0))

visualize(lrp_result, "uniform_epsilon_pso_result")


