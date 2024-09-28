<div align="center">    
 
# Explainable Schizophrenia Classification from rs-fMRI using SwiFT and Swarm-LRP

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.7+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>

</div>


## 🧠&nbsp;&nbsp;Introduction
Schizophrenia is a disease characterized by complex alterations in brain structure and function. It presents significant challenges for identification using neuroimaging methods like fMRI, principally due to the clinical necessity of model explainability. Existing machine learning approaches, while achieving high degrees of accuracy, have not been widely adopted in clinical settings due to their lack of transparency. To address this, we build upon the Swin 4D fMRI Transformer model through the implementation of transformer layer-wise relevance propagation. We train this model to classify resting-state fMRI data from schizophrenia patients and controls. Additionally, we propose Swarm-LRP, a novel particle swarm optimization method for layer-wise relevance propagation rules utilizing Quantus explainable AI toolkit metrics. TransLRP demonstrates improved interpretability, outperforming benchmark model-agnostic explanations. These findings suggest that we can achieve a meaningfully interpretable representation of resting-state brain activity associated with schizophrenia diagnosis.


> Effective usage of this repository requires learning a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai).

The core classes are as follows.
- `COBREDataModule`: A class that encapsulates all the steps needed to process data.
- `LitClassifier`: A class that encapsulates the following things: model, train & valid & test steps, and optimizers.
- `pl.Trainer`: A class that contains all of the other processes to operate the `LitClassifier`.
- `QuantusEval`: A class that holds utility methods for running metric computations
---

## 1. Description
This repository implements Swarm-LRP, LRP-epsilon, LRP-gamma, and utilizes [SwiFT](http:/https://github.com/Transconnectome/SwiFT/tree/main/project/ "SwiFT") to generate classifications. 

- Our code has the following
	- From the SwiFT Repository and adapted for our data from [COBRE](httphttps://fcon_1000.projects.nitrc.org/indi/retro/cobre.html:// "COBRE").
  - Trainer based on PyTorch Lightning for running SwiFT.
  - `SwinTransformer4D` architecture and its variants
  - Data preprocessing/loading pipelines for 4D fMRI datasets.
  - Self-supervised learning strategies
	- The [Quantus](http://https://github.com/understandable-machine-intelligence-lab/Quantus "Quantus") XAI framework adapted for 4D fMRI volumes
	- Swarm-LRP 
		- Particle swarm implementations for LRP
		- Testing framework to run baselines and our TransLRP code


## 2. How to install
We highly recommend you to use our conda environment.
```bash
# install project from zip
cd SwiFT-LRP-FULL
conda env create -f envs/py39.yaml
conda activate swift
 ```

## 3. Project Structure
Our directory structure looks like this:

```
├── COBRE_MNI_to_TRs                   <- Dataset
├── output                       <- Experiment log and checkpoints will be saved here once you train a model - best checkpoints are located here
├── envs                         <- Conda environment
├── pretrained_models            <- Pretrained model checkpoints
│   ├── split_hcp.txt                      <- Data split for the trained HCP model 
├── project                 
│   ├── module                   <- Every module is given in this directory
│   │   ├── models               <- Models (Swin fMRI Transformer + TransLRP)
│   │   ├── utils                  <- Utilities (including fMRI volume visualization)
│   │   │    ├── data_module.py  <- Dataloader & codes for matching fMRI scans and target variables
│   │   │    └── data_preprocessing_and_load
│   │   │        ├── datasets.py           <- Dataset Class for each dataset
│   │   │        └── preprocessing.py      <- Preprocessing codes for step 6
│   │   ├── pl_classifier.py    <- LightningModule
│.   │  └── ExplanationGenerator.py    <- Driver class that generates TransLRP for Swift
│   └── main.py                 <- Main code that trains and tests the 4DSwinTransformer model
│
├── test                 <- Code for Debugging SwiFT
│ 
├── sample_scripts              <- Example shell scripts for training (more thorough than readme)
├── project                      <- Quantus package with alterations for 4D fMRI
├── Swarm-LRP 
│   │   ├── quantusEval.py               <- Utilities Class
│   │   ├── particleSwarm.py            <- Main PSO function
│   │   ├── swift_ig.py                      <- Testiing code for baseline Integrated Gradients
│   │   ├── explanation_stats.py       <- Generates Mean and Stdev statistics for test loader evaluated on a particular explanation method

├── .gitignore                  <- List of files/folders ignored by git
├── export_DDP_vars.sh          <- setup file for running torch DistributedDataParallel (DDP) 
└── README.md
```

<br>
## 4. Running experiments
### 4.1 Arguments for trainer
You can check the arguments list by using -h
 ```bash
python project/main.py --data_module dummy --classifier_module default -h
```


### 4.2 Commands/scripts for running classification/regression tasks
- Sample script calls are in sample_scripts/sample_script.sh with more detail
- Append ```2>&1| tee <filename>.txt``` to pipe output to a text file


## 5. Pretrained model checkpoints
We provide some pretrained model checkpoints under the output/default directory. Swift pretrained model checkpoints are under the pretrained_models directory.
* current_best.ckpt is Checkpoint 2 from the paper with proper preprocessing
- last_best.ckpt is Checkpoint 1
* To fine-tune the provided models for another task, use the load_model_path argument on main.py


