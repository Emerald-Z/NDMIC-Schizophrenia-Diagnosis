cd .. # move to where 'SwiFT is located'
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate swift
 
TRAINER_ARGS='--accelerator gpu --max_epochs 5 --precision 16 --num_nodes 1 --devices 1 --strategy DDP --log_every_n_steps 25' 
MAIN_ARGS='--loggername tensorboard --classifier_module v6 --dataset_name COBRE --image_path /raid/neuro/Schizophrenia_Project/NDMIC-Schizophrenia-Diagnosis/SwiFT/COBRE_MNI_to_TRs --explanation_method TransLRP' 
DATA_ARGS='--batch_size 8 --input_type rest --num_workers 8'
DEFAULT_ARGS='--project_name runs'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task schizo' 
LRP_ARGS='--test_only --lrp_test --load_model_path /raid/neuro/Schizophrenia_Project/NDMIC-Schizophrenia-Diagnosis/SwiFT-LRP-Full/output/default/current_best.ckpt' #TODO: replace these again
RESUME_ARGS=''

export NEPTUNE_API_TOKEN="{neptune API token}" # when using neptune as a logger

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# runs the main file for interactive training
# LRP args will dictate if you train or run LRP on an instance
python project/main.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1

# Driver file for running Swarm-LRP
# modify function calls in main method for now
python project/Swarm-LRP/quantusEval.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1 

# runs the explanation stats file to compute averages and standard deviations across a split for a particular model
# --explanation_model lrp or ig
python project/Swarm-LRP/explanation_stats.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1

python project/Swarm-LRP/mp_test.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1

python project/Swarm-LRP/batch_test.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1

python project/Swarm-LRP/quantusEval.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $DEFAULT_ARGS $OPTIONAL_ARGS $LRP_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 96 96 96 20 --patch_size 6 6 6 1