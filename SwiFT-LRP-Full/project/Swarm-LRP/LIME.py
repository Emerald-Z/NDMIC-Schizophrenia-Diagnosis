import torch
import torch.nn.functional as F

#from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso

from torchvision.models import resnet18
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as T
from captum.attr._core.lime import get_exp_kernel_similarity_function

from visualization import visualize

from PIL import Image
import matplotlib.pyplot as plt

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pytorch_lightning as pl
import numpy as np
# from module import LitClassifier
# import neptune.new as neptune
import sys
sys.path.insert(0, '/raid/neuro/Schizophrenia_Project/NDMIC-Schizophrenia-Diagnosis/SwiFT-LRP-Full') # replace path later
from project.module.utils.data_module import fMRIDataModule
from project.module.pl_classifier import LitClassifier
from utils import generate_feature_mask

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

kwargs = {
    "nt_samples": 5,
    "nt_samples_batch_size": 5,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 5,
}

# voc_ds = VOCSegmentation(
#     './VOC',
#     year='2012',
#     image_set='train',
#     download=False,
#     transform=T.Compose([
#         T.ToTensor(),
#         T.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )    
#     ]),
#     target_transform=T.Lambda(
#         lambda p: torch.tensor(p.getdata()).view(1, p.size[1], p.size[0])
#     )
# )

# It is time to configure our Lime algorithm now. Essentially, Lime trains an interpretable surrogate model to simulate the target model's predictions. So, building an appropriate interpretable model is the most critical step in Lime. Fortunately, Captum has provided many most common interpretable models to save the efforts. We will demonstrate the usages of Linear Regression and Linear Lasso. Another important factor is the similarity function. Because Lime aims to explain the local behavior of an example, it will reweight the training samples according to their similarity distances. By default, Captum's Lime uses the exponential kernel on top of the consine distance. We will change to euclidean distance instead which is more popular in vision. 
exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

lr_lime = Lime(
    model, 
    interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
    similarity_func=exp_eucl_distance
)


# Next, we will analyze these groups' influence on the most confident prediction `television`. Every time we call Lime's `attribute` function, an interpretable model is trained around the given input, so unlike many other Captum's attribution algorithms, it is strongly recommended to only provide a single example as input (tensors with first dimension or batch size = 1). There are advanced use cases of passing batched inputs. Interested readers can check the [documentation](https://captum.ai/api/lime.html) for details.
# 
# In order to train the interpretable model, we need to specify enough training data through the argument `n_samples`. Lime creates the perturbed samples in the form of interpretable representation, i.e., a binary vector indicating the “presence” or “absence” of features. Lime needs to keep calling the target model to get the labels/values for all perturbed samples. This process can be quite time-consuming depending on the complexity of the target model and the number of samples. Setting the `perturbations_per_eval` can batch multiple samples in one forward pass to shorten the process as long as your machine still has capacity. You may also consider turning on the flag `show_progress` to display a progess bar showing how many forward calls are left.
def generate_feature_mask():
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

input_data_single = input_data_single / torch.sum(input_data_single)
zero_target = torch.from_numpy(np.array([0])).to(device)
print("shapes: ", x_batch.device, zero_target.device)
label_idx = y_batch
attrs = lr_lime.attribute(
    x_batch,
    target=zero_target,
    feature_mask=generate_feature_mask().to(device),
    n_samples=40,
    perturbations_per_eval=16,
    show_progress=True
).squeeze(0)

print("result calculated: ", attrs)
visualize(attrs, "lime_result")
# print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())


# Now, let us use Captum's visualization tool to view the attribution heat map.

# def show_attr(attr_map):
#     viz.visualize_image_attr(
#         attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels 
#         method='heat_map',
#         sign='all',
#         show_colorbar=True
#     )
    
# show_attr(attrs)


# The result looks decent: the television segment does demonstrate strongest positive correlation with the prediction, while the chairs has relatively trivial impact and the border slightly shows negative contribution.
# 
# However, we can further improve this result. One desired characteristic of interpretability is the ease for human to comprehend. We should help reduce the noisy interference and emphisze the real influential features. In our case, all features more or less show some influences. Adding lasso regularization to the interpretable model can effectively help us filter them. Therefore, let us try Linear Lasso with a fit coefficient `alpha`. For all built-in sklearn wrapper model, you can directly pass any sklearn supported arguments.
# 
# Moreover, since our example only has 4 segments, there are just 16 possible combinations of interpretable representations in total. So we can exhaust them instead random sampling. The `Lime` class's argument `perturb_func` allows us to pass a generator function yielding samples. We will create the generator function iterating the combinations and set the `n_samples` to its exact length.

# n_interpret_features = len(seg_ids)

# def iter_combinations(*args, **kwargs):
#     for i in range(2 ** n_interpret_features):
#         yield torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]).unsqueeze(0)
    
# lasso_lime = Lime(
#     resnet, 
#     interpretable_model=SkLearnLasso(alpha=0.08),
#     similarity_func=exp_eucl_distance,
#     perturb_func=iter_combinations
# )

# attrs = lasso_lime.attribute(
#     img.unsqueeze(0),
#     target=label_idx,
#     feature_mask=feature_mask.unsqueeze(0),
#     n_samples=2 ** n_interpret_features,
#     perturbations_per_eval=16,
#     show_progress=True
# ).squeeze(0)

# print('Attribution range:', attrs.min().item(), 'to', attrs.max().item())
# show_attr(attrs)