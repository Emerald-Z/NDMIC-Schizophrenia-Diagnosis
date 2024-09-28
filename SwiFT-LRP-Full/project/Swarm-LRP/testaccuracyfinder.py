import numpy as np

# import quantus
import torch
import torch.multiprocessing as tmp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from torchmetrics.classification import BinaryAccuracy

import torch
from collections import OrderedDict
import pytorch_lightning as pl
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
import quantus

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

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

labels = {"patch_em": {"l": "epsilon"}, "drop": "gamma", "pos_emb": {"add1": "alpha", "add2": "alpha"},
       "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "gamma", "do": "alpha", "l2": "epsilon", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
       "norm": "alpha", "mlp": {"l1": "epsilon", "l2": "gamma", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"},
       "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"},
       "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"},
       "norm": "alpha", "mlp": {"l1": "epsilon", "l2": "gamma", "gelu": "alpha", "do": "alpha"},
       "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "gamma",
       "clf_mlp": {"l": "zero", "sm": "alpha", "do": "alpha"}}

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

LRP_PARAMS = [1.0000000e+00, -3.6812282e-08,  6.2181394e-01]

'''
    Copied from quantusTest - i didn't do imports because it it runs teh stuff in the file but
    we can think abou tmaking everythign a class when we submit code
'''
def evaluate_metric(metric_init, input, x, y, lrp_result):
    return metric_init(model=model, 
                        x_batch=input,
                        y_batch=y,
                        a_batch=lrp_result,
                        device=device,
                        explain_func=quantus.explain,
                        explain_func_kwargs={"method": "TransLRP", "softmax": False, "alpha": x[0], "epsilon": x[1], "gamma": x[2]})[0]
                        # explain_func_kwargs={"method": "Trans", "softmax": False})[0]

    
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

    # print("faithfulness: ", results[0], "complexity: ", results[1], "sensitivity: ", results[2])
    return np.array(results)


# Fetch data loader
data_loader = iter(data_module.test_dataloader())
dataset = data_module.test_dataloader().dataset
num_samples = len(dataset)
metric_values = np.zeros((int(num_samples/7), 3))
metric_labels = np.zeros((int(num_samples/7), 2))
subject_idx = 0
subjects = []

# integrated_gradients  = IntegratedGradients(model)
# noise_tunnel = NoiseTunnel(integrated_gradients)

# kwargs = {
#     "nt_samples": 5,
#     "nt_samples_batch_size": 5,
#     "nt_type": "smoothgrad_sq", # 1
#     #"stdevs": 0.05,
#     "internal_batch_size": 5,
# }

correct_num = 0
all_sum = 0
num = 0
for (data_idx, single_data) in enumerate(data_loader):
    input_data = single_data["fmri_sequence"]
    input_labels = single_data["schizo"]
    # print("input: ", input_labels)
    # print subject name
    for i in range(input_labels[0].shape[0]):
        # Select the first item in the batch for LRP analysis
        # Assuming that your model and the data_loader support processing single items directly
        # input_data_single = input_data.to(device)  # Ensure input has batch dimension if needed input_data[0].unsqueeze(0).to(device)
        curr_subj = single_data["subject_name"][i]
        
        input_data_single = input_data[i].unsqueeze(0).to(device)
        output = torch.sigmoid(model.forward(input_data_single))
        all_sum += output.item()
        num += 1
        if int(output.item()) >= .469:
            if input_labels[0][i].item() == 1:
                correct_num += 1
        else:
            if input_labels[0][i].item() == 0:
                correct_num += 1
        print("outside output:", output)
        if curr_subj not in subjects:

            #print(curr_subj)
            input_data_single = input_data[i].unsqueeze(0).to(device)
            # Determine the predicted class or the index for LRP
            # predicted_index = output.argmax(dim=1).item()  # Get the index of the max logit which is the predicted class
            
            #output = model.forward(input_data_single)
            
            #output = torch.sigmoid(output)
            acc_func = BinaryAccuracy().to(output.device)
            #print(torch.tensor([input_labels[0][i]]).to(device))
            #print((output >= 0).int()[0])
            acc = acc_func((output >= 0).int()[0], torch.tensor([input_labels[0][i]]).to(device))
            #print(curr_subj, "prediction:", ("correct" if int(acc.item()) else "WRONG!"), "true value:", input_labels[0][i])
            print("true value:", input_labels[0][i], "output:", output[0])
            #index = 1 if output.cpu().data.numpy()[0][0] > 0 else 0
            
            x_batch = input_data_single
            # y_batch = input_labels[0].to(device) # [tensor]
            y_batch = input_labels[0][i].to(device) # [tensor]
            #metric_labels[subject_idx, 0] = y_batch
            #metric_labels[subject_idx, 1] = index
            #print(y_batch, index)
            
            subject_idx += 1
            subjects.append(curr_subj)
    
        # print("LRP result:", lrp_result.shape)
        # visualize(lrp_result)
print(correct_num)
print(all_sum)
print(num)
# generate translrp explanation
# a_batch_saliency = quantus.normalise_func.normalise_by_negative(Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())

# Save x_batch and y_batch as numpy arrays that will be used to call metric instances.

# print(torch.sum(input_data_single))
# print(torch.sum(lrp_result))


# Quick assert.
# assert [isinstance(obj, np.ndarray) for obj in [x_batch, y_batch, a_batch_saliency, a_batch_intgrad]]

