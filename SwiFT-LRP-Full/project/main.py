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

import ast

def cli_main():

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
    parser.add_argument("--labels", type=str, help="LRP labels dict")

    temp_args, _ = parser.parse_known_args()

    # Set classifier
    Classifier = LitClassifier
    
    # Set dataset
    Dataset = fMRIDataModule
    print("DATASET CREATED")
    
    # add two additional arguments
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    torch.use_deterministic_algorithms(True) # FIXME: needed?
    
    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
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
    labels = {"patch_em": {"l": "alpha"}, "drop": "alpha", "pos_emb": {"add1": "alpha", "add2": "alpha"}, 
    "basic_layer": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
    "norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"}, "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, 
    "full_att": {"block": {"norm1": "alpha", "win_attn": {"l1": "alpha", "do": "alpha", "l2": "alpha", "sm": "alpha", "add": "alpha", "ein1": "alpha", "ein2": "alpha"}, 
    "norm": "alpha", "mlp": {"l1": "alpha", "l2": "alpha", "gelu": "alpha", "do": "alpha"},
    "add1": "alpha", "add2": "alpha", "clone1": "alpha", "clone2": "alpha"}, "down": {"l": "alpha", "norm": "alpha"}, "ein": "alpha"}, "norm": "alpha", "avg_pool": "alpha", "l": "alpha",
    "clf_mlp": {"l": "alpha", "sm": "alpha", "do": "alpha"}}

    args.labels = labels

    # ------------ data -------------
    print("ARGS: ", args)
    data_module = Dataset(**vars(args))

    # ------------ logger -------------
    if args.loggername == "tensorboard":
        # logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
        print(dirpath)
        logger = TensorBoardLogger(dirpath)
    # elif args.loggername == "neptune":
    #     API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
    #     # project_name should be "WORKSPACE_NAME/PROJECT_NAME"
    #     run = neptune.init(api_token=API_KEY, project=args.project_name, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, run=exp_id)
        
    #     if exp_id == None:
    #         setattr(args, "id", run.fetch()['sys']['id'])

    #     logger = NeptuneLogger(run=run, log_model_checkpoints=False)
    #     dirpath = os.path.join(args.default_root_dir, logger.version)
    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    # callback for pretraining task
    if args.pretraining:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_loss",
            filename="checkpt-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            mode="min",
        )
    # callback for classification task
    elif args.downstream_task == "schizo" or args.downstream_task == "Dummy" or args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
    # callback for regression task
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.2f}",
            save_last=True,
            mode="min",
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # ------------ trainer -------------
    print("TRAINER")
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
        )
    else:
        print('not using gradient clipping')
        print(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            #val_check_interval=100 if not args.scalability_check else None,
            callbacks=callbacks
        )

    # ------------ model -------------
    print("MODEL")
    model = Classifier(data_module = data_module, **vars(args)) 

    if args.load_model_path is not None:
        print(f'loading model from {args.load_model_path}')
        path = args.load_model_path
        ckpt = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                new_state_dict[k.removeprefix("model.")] = v
        model.model.load_state_dict(new_state_dict)

    if args.freeze_feature_extractor:
        # layers are frozen by using eval()
        model.model.eval()
        # freeze params
        for name, param in model.model.named_parameters():
            if 'output_head' not in name: # unfreeze only output head
                param.requires_grad = False
                print(f'freezing layer {name}')

    # ------------ run -------------
    if args.test_only:
        if args.lrp_test:
            print("Calculating LRP...")
            lrp_instance = LRP(model)
            # Fetch data loader
            data_loader = iter(data_module.test_dataloader())
            for single_data in data_loader:
                input_data = single_data["fmri_sequence"]
                # Select the first item in the batch for LRP analysis
                # Assuming that your model and the data_loader support processing single items directly
                print("INPUT DATA SHAPE:", input_data.shape)
                input_data_single = input_data[0].unsqueeze(0)  # Ensure input has batch dimension if needed
                print("INPUT DATA SHAPE:", input_data_single.shape)
                # Determine the predicted class or the index for LRP
                # predicted_index = output.argmax(dim=1).item()  # Get the index of the max logit which is the predicted class
            
                # Execute LRP
                lrp_result = lrp_instance.generate_LRP(input_data_single, alpha=1, epsilon=0.75, gamma=0.75)
                print("LRP result:", lrp_result)
                visualize(lrp_result)
                break # should only do one data point (could edit this to do them all)
        else:
            trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
