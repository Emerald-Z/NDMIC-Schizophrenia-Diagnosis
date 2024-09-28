import numpy as np
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pytorch_lightning as pl
import sys
sys.path.insert(0, '/raid/neuro/Schizophrenia_Project/NDMIC-Schizophrenia-Diagnosis/SwiFT-LRP-Full') # replace path later

from project.module.utils.data_module import fMRIDataModule
from project.module.pl_classifier import LitClassifier

# import LRP class
from project.module.Explanation_generator import LRP
from utils import generate_feature_mask
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum._utils.models.linear_model import SkLearnLinearRegression

from quantusEval import QuantusEval
from particleSwarm import pso
import quantus
import time
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# ------------ args -------------
parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
parser.add_argument("--dataset_name", default="COBRE", type=str, choices=["S1200", "ABCD", "UKB", "Dummy", "COBRE"])
parser.add_argument("--downstream_task", type=str, default="schizo", help="downstream task")
parser.add_argument("--downstream_task_type", type=str, default="default", help="select either classification or regression according to your downstream task")
parser.add_argument("--classifier_module", default="v6", type=str, help="A name of lightning classifier module (outdated argument)")
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
       
labels = {"patch_em": {"l": "gamma"}, "drop": "gamma", "pos_emb": {"add1": "gamma", "add2": "gamma"},
       "basic_layer": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
       "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"}, "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"},
       "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"},
       "full_att": {"block": {"norm1": "gamma", "win_attn": {"l1": "gamma", "do": "gamma", "l2": "gamma", "sm": "gamma", "add": "gamma", "ein1": "gamma", "ein2": "gamma"},
       "norm": "gamma", "mlp": {"l1": "gamma", "l2": "gamma", "gelu": "gamma", "do": "gamma"},
       "add1": "gamma", "add2": "gamma", "clone1": "gamma", "clone2": "gamma"}, "down": {"l": "gamma", "norm": "gamma"}, "ein": "gamma"}, "norm": "gamma", "avg_pool": "gamma", "l": "gamma",
       "clf_mlp": {"l": "gamma", "sm": "gamma", "do": "gamma"}}

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

LRP_PARAMS = [2, 0.0000001, 0.3]
evaluator = QuantusEval(model)

# Fetch data loader
data_loader = iter(data_module.test_dataloader())
dataset = data_module.test_dataloader().dataset
num_samples = len(dataset)

metric_values = np.zeros((23, 3))

single_metric = np.zeros((23, 1))
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

eff_complex = quantus.EffectiveComplexity(
    eps=1e-7,
    disable_warnings=True,
    normalise=True,
    abs=True)

faith_corr = quantus.FaithfulnessCorrelation(
    similarity_func=quantus.similarity_func.correlation_pearson,
    nr_runs=50,
    subset_size= 2240,
    abs=True,
    normalise=True,
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    perturb_baseline="black"
)

if __name__ == "__main__":
    for (data_idx, single_data) in enumerate(data_loader):
        input_data = single_data["fmri_sequence"]
        input_labels = single_data["schizo"]

        curr_subj = single_data["subject_name"]
        if curr_subj not in subjects:
            print(curr_subj)
            input_data_single = input_data.to(device)
            # Determine the predicted class or the index for LRP
            # predicted_index = output.argmax(dim=1).item()  # Get the index of the max logit which is the predicted class
            
            output = model.forward(input_data_single)
            print(output, input_labels)
            x_batch = input_data_single
            # y_batch = input_labels[0].to(device) # [tensor]
            y_batch = input_labels # [tensor]
            input_data_single = input_data_single[:4]
            print(input_data_single.shape)
            if method == "TransLRP":
                # Execute LRP
                lrp_result = lrp_instance.generate_LRP(input_data_single, alpha=LRP_PARAMS[0], epsilon=LRP_PARAMS[1], gamma=LRP_PARAMS[2], device=device) # there's a way to pass them all in **
                lrp_result = lrp_result.cpu().unsqueeze(0).detach().numpy()
            elif method == "IntegratedGradients":
                # IG
                lrp_result = noise_tunnel.attribute(input_data_single, baselines=input_data_single[0,0,0,0,0,0].item(),target=None,**kwargs)
                lrp_result = lrp_result.cpu().detach().numpy()
            elif method == "Saliency":
                lrp_result = quantus.explain(model, x_batch,  np.array([0]), method="Saliency")
            elif method == "Lime":
                lrp_result = quantus.explain(model, x_batch,  np.array([0]), method="Lime", feature_mask=generate_feature_mask().to(device),
                                                interpretable_model=SkLearnLinearRegression())
                y_batch = torch.from_numpy(np.array([0]))
            else:
                raise ValueError(
                        f"'Method {method} not recognized"
                    )

            x_batch, y_batch = x_batch.cpu().numpy(), np.array(y_batch)
            lrp_result = lrp_result.swapaxes(0,1)
            print("attr shape: ", lrp_result.shape, y_batch.shape)
            y_batch = y_batch.squeeze(0)[:4]
            # normalize lrp result
            row_sums = lrp_result.sum(axis=(1, 2, 3, 4, 5), keepdims=True)
            lrp_result /= row_sums

            input_data_single = input_data_single[:4]
            row_sums = input_data_single.sum(dim=(1, 2, 3, 4, 5), keepdim=True)
            input_data_single /= row_sums

            # metric_values[subject_idx, :] =  evaluator.evaluate_metrics(LRP_PARAMS, y_batch, input_data_single, lrp_result, method)
            # print(metric_values[subject_idx, :])
            # complex = evaluator.evaluate_metrics_batch(LRP_PARAMS, y_batch, input_data_single, lrp_result, method)
            # print(complex)
            # metric_values[subject_idx] = complex #/ np.prod(input_data_single.numel())
            subject_idx += 1
            subjects.append(curr_subj)
        break

    caller = QuantusEval(model)
    print("****** BASELINE ******")
    baselines = caller.evaluate_metrics_batch([1, 0.0000001, 0.75], y_batch, input_data_single, lrp_result, "TransLRP")
    
    print("******* PSO ******")
    # Run the PSO algorithm 
    
    # No override (value)
    solution, fitness = pso(caller.optimizer_single, dim=1, num_particles=5, 
                            max_iter=5, input=input_data_single, y=y_batch, baselines=baselines)

    print('It took', time.time()-start, 'seconds.')
    # print("mean : ", np.mean(metric_values, 0))
    # print("standard dev", np.std(metric_values, 0))