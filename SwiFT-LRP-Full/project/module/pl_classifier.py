import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy

import torchmetrics
import torchmetrics.classification
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from torchmetrics import  PearsonCorrCoef # Accuracy,
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb


from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.losses import NTXentLoss, global_local_temporal_contrastive
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

from einops import rearrange

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class LitClassifier(pl.LightningModule):
    def __init__(self, data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
       
        # you should define target_values at the Dataset classes
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax': 
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler
        self.labels = kwargs["labels"]
        self.model = load_model(self.hparams.model, labels=self.labels, hparams=self.hparams)

        # Heads
        if not self.hparams.pretraining:
            if self.hparams.downstream_task == 'schizo' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                self.output_head = load_model("clf_mlp", labels=self.labels["clf_mlp"], hparams=self.hparams)
            elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                self.output_head = load_model("reg_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams) # TODO: do labels for output head
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()

        if self.hparams.adjust_thresh:
            self.threshold = 0

    def forward(self, x):
        return self.output_head(self.model(x))
    
    def relprop(self, cam, **kwargs):
        cam = self.output_head.relprop(cam, **kwargs)
        cam = self.model.relprop(cam, **kwargs)
        return cam
    
    def augment(self, img):

        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img
    
    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, tr, schizo = batch.values()
        #fmri, subj, target_value, tr, schizo = batch.values()
       
        if augment_during_training:
            fmri = self.augment(fmri)

        feature = self.model(fmri)

        # Classification task
        if self.hparams.downstream_task == 'schizo' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            logits = self.output_head(feature).squeeze() #self.clf(feature).squeeze()
            target = torch.stack(schizo, dim=0).squeeze()
        # Regression task
        elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
            # target_mean, target_std = self.determine_target_mean_std()
            logits = self.output_head(feature) # (batch,1) or # tuple((batch,1), (batch,1))
            # unnormalized_target = target_value.float() # (batch,1)

            # if self.hparams.label_scaling_method == 'standardization': # default
            #     target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            # elif self.hparams.label_scaling_method == 'minmax':
            #     target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
            
        return subj, logits, target
    
    def _calculate_loss(self, batch, mode):
        if self.hparams.pretraining:
            fmri, subj, tr, schizo = batch.values()
            #fmri, subj, target_value, tr, sex = batch.values()
            
            cond1 = (self.hparams.in_chans == 1 and not self.hparams.with_voxel_norm)
            assert cond1, "Wrong combination of options"
            loss = 0

            if self.hparams.use_contrastive:
                assert self.hparams.contrastive_type != "none", "Contrastive type not specified"

                # B, C, H, W, D, T = image shape
                y, diff_y = fmri

                batch_size = y.shape[0]
                if (len(subj) != len(tuple(subj))) and mode == 'train':
                    print('Some sub-sequences in a batch came from the same subject!')
                criterion = NTXentLoss(device='cuda', batch_size=batch_size,
                                        temperature=self.hparams.temperature,
                                        use_cosine_similarity=True).cuda()
                criterion_ll = NTXentLoss(device='cuda', batch_size=2,
                                            temperature=self.hparams.temperature,
                                            use_cosine_similarity=True).cuda()
                
                # type 1: IC
                # type 2: LL
                # type 3: IC + LL
                if self.hparams.contrastive_type in [1, 3]:
                    out_global_1 = self.output_head(self.model(self.augment(y)),"g")
                    out_global_2 = self.output_head(self.model(self.augment(diff_y)),"g")
                    ic_loss = criterion(out_global_1, out_global_2)
                    loss += ic_loss

                if self.hparams.contrastive_type in [2, 3]:
                    out_local_1 = []
                    out_local_2 = []
                    out_local_swin1 = self.model(self.augment(y))
                    out_local_swin2 = self.model(self.augment(y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    out_local_swin1 = self.model(self.augment(diff_y))
                    out_local_swin2 = self.model(self.augment(diff_y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    ll_loss = 0
                    # loop over batch size
                    for i in range(out_local_1[0].shape[0]):
                        # out_local shape should be: BS, n_local_clips, D
                        ll_loss += criterion_ll(torch.stack(out_local_1, dim=1)[i],
                                                torch.stack(out_local_2, dim=1)[i])
                    loss += ll_loss

                result_dict = {
                    f"{mode}_loss": loss,
                }        
        else:
            subj, logits, target = self._compute_logits(batch, augment_during_training = self.hparams.augment_during_training)

            if self.hparams.downstream_task == 'schizo' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                loss = F.binary_cross_entropy_with_logits(logits, target.float()) # target is float
                acc = self.metric.get_accuracy_binary(logits, target.float().squeeze())
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_acc": acc,
                }

            elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                loss = F.mse_loss(logits.squeeze())
                l1 = F.l1_loss(logits.squeeze(), target.squeeze())
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_mse": loss,
                    f"{mode}_l1_loss": l1
                }
        self.log_dict(result_dict, prog_bar=True, sync_dist=True, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size) # batch_size = batch_size
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode):
        # (total iteration/world_size) numbers of samples are passed into _evaluate_metrics.
        subjects = np.unique(subj_array)
        
        subj_avg_logits = []
        subj_targets = []
        for subj in subjects:
            subj_logits = total_out[subj_array == subj,0] 
            subj_avg_logits.append(torch.mean(subj_logits).item())
            subj_targets.append(total_out[subj_array == subj,1][0].item())
        subj_avg_logits = torch.tensor(subj_avg_logits, device = total_out.device) 
        subj_targets = torch.tensor(subj_targets, device = total_out.device) 
        
    
        if self.hparams.downstream_task == 'schizo' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            if self.hparams.adjust_thresh:
                # move threshold to maximize balanced accuracy
                best_bal_acc = 0
                best_thresh = 0
                for thresh in np.arange(-5, 5, 0.01):
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=thresh).int().cpu())
                    if bal_acc > best_bal_acc:
                        best_bal_acc = bal_acc
                        best_thresh = thresh
                print("BEST THRESHOLD:", best_thresh)
                self.log(f"{mode}_best_thresh", best_thresh, sync_dist=True)
                self.log(f"{mode}_best_balacc", best_bal_acc, sync_dist=True)
                fpr, tpr, thresholds = roc_curve(subj_targets.cpu(), subj_avg_logits.cpu())
                idx = np.argmax(tpr - fpr)
                youden_thresh = thresholds[idx]
                acc_func = BinaryAccuracy().to(total_out.device)
                self.log(f"{mode}_youden_thresh", youden_thresh, sync_dist=True)
                self.log(f"{mode}_youden_balacc", balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=youden_thresh).int().cpu()), sync_dist=True)

                if mode == 'valid':
                    self.threshold = youden_thresh
                elif mode == 'test':
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=self.threshold).int().cpu())
                    self.log(f"{mode}_balacc_from_valid_thresh", bal_acc, sync_dist=True)
            else:
                acc_func = BinaryAccuracy().to(total_out.device)
                
            auroc_func = BinaryAUROC().to(total_out.device)
            acc = acc_func((subj_avg_logits >= 0).int(), subj_targets)
            bal_acc_sk = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=0).int().cpu())
            auroc = auroc_func(torch.sigmoid(subj_avg_logits), subj_targets)

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_balacc", bal_acc_sk, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

        # regression target is normalized
        elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':          
            mse = F.mse_loss(subj_avg_logits, subj_targets)
            mae = F.l1_loss(subj_avg_logits, subj_targets)
            
            # reconstruct to original scale
            if self.hparams.label_scaling_method == 'standardization': # default
                adjusted_mse = F.mse_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                adjusted_mae = F.l1_loss(subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0], subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                adjusted_mse = F.mse_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                adjusted_mae = F.l1_loss(subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0], subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
            pearson = PearsonCorrCoef().to(total_out.device)
            prearson_coef = pearson(subj_avg_logits, subj_targets)
            
            self.log(f"{mode}_corrcoef", prearson_coef, sync_dist=True)
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)
            self.log(f"{mode}_adjusted_mse", adjusted_mse, sync_dist=True) 
            self.log(f"{mode}_adjusted_mae", adjusted_mae, sync_dist=True) 

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.hparams.pretraining:
            if dataloader_idx == 0:
                self._calculate_loss(batch, mode="valid")
            else:
                self._calculate_loss(batch, mode="test")
        else:
            subj, logits, target = self._compute_logits(batch)
            if self.hparams.downstream_task_type == 'multi_task':
                output = torch.stack([logits[1].squeeze(), target], dim=1) # logits[1] : regression head
            else:
                output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)
            return (subj, output)

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        if not self.hparams.pretraining:
            outputs_valid = outputs[0]
            outputs_test = outputs[1]
            subj_valid = []
            subj_test = []
            out_valid_list = []
            out_test_list = []
            for subj, out in outputs_valid:
                subj_valid += subj
                out_valid_list.append(out.detach())
            for subj, out in outputs_test:
                subj_test += subj
                out_test_list.append(out.detach())
            subj_valid = np.array(subj_valid)
            subj_test = np.array(subj_test)
            total_out_valid = torch.cat(out_valid_list, dim=0)
            total_out_test = torch.cat(out_test_list, dim=0)

            # save model predictions if it is needed for future analysis
            # self._save_predictions(subj_valid,total_out_valid,mode="valid")
            # self._save_predictions(subj_test,total_out_test, mode="test") 
            
            # Calculate the percentage of matching pairs for test outputs
            predictions = (total_out_test[:, 0] >= 0).float()
            targets = total_out_test[:, 1]
            matching_pairs = (predictions == targets).sum().item()
            total_pairs = len(targets)
            matching_percentage = (matching_pairs / total_pairs) * 100
            print(f"\nTEST ACCURACY: {matching_percentage:.2f}%\n")
                
            # evaluate 
            self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
            self._evaluate_metrics(subj_test, total_out_test, mode="test")
            
    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'schizo':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1
        
        if self.hparams.strategy == None : 
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            total_subj_accuracy = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks     
            accuracy_dict = {}
            for dct in total_subj_accuracy:
                for subj, metric_dict in dct.items():
                    if subj not in accuracy_dict:
                        accuracy_dict[subj] = metric_dict
                    else:
                        accuracy_dict[subj]['score']+=metric_dict['score']
                        accuracy_dict[subj]['count']+=metric_dict['count']
            self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved) 
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)

    def test_step(self, batch, batch_idx):
        subj, logits, target = self._compute_logits(batch)
        output = torch.stack([logits.squeeze(), target.squeeze()], dim=1)#torch.stack([logits.squeeze(), target.squeeze()], dim=1)
        return (subj, output)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            subj_test = [] 
            out_test_list = []
            for subj, out in outputs:
                subj_test += subj
                out_test_list.append(out.detach())
            subj_test = np.array(subj_test)
            total_out_test = torch.cat(out_test_list, dim=0)
            # self._save_predictions(subj_test, total_out_test, mode="test") 
            self._evaluate_metrics(subj_test, total_out_test, mode="test")
    
    def on_train_epoch_start(self) -> None:
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings=np.zeros((self.repetitions,1))
        return super().on_train_epoch_start()
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx-self.gpu_warmup] = curr_time
            elif (batch_idx-self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)
                
                Throughput = (self.repetitions*self.hparams.batch_size*int(self.hparams.num_nodes) * int(self.hparams.devices))/self.total_time
                
                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:',mean_syn)
                print('std_syn:',std_syn)
                
        return super().on_train_batch_end(out, batch, batch_idx)


    # def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")
        
        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05) # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1
            
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:',self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")
        
        # pretraining-related
        group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int, help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")
        
        # model related
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int, help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")

        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")
        
        return parser
