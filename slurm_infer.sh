#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=bench-405b-long           # Name of the job
#SBATCH --output=output-%j.log            # Output log file
#SBATCH --error=error-%j.log              # Error log file
#SBATCH --time=4:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi3008x               # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda deactivate
conda activate medusa-new
cd /work1/deming/seliny2/Medusa
export HOME=/work1/deming/shared
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:/lib
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/work1/deming/seliny2/Medusa
ulimit -n 65536
# accelerate launch --config_file /work1/deming/shared/.cache/huggingface/accelerate/default_config.yaml  \
# gen_model_answer_medusa_legacy.py --model-path FasterDecoding/medusa-vicuna-7b-v1.3 \
# --model-id lmsys/vicuna-7b-v1.3
# eval: 
# accelerate launch --config_file /work1/deming/shared/.cache/huggingface/accelerate/default_config.yaml  \
# gen_model_answer_medusa_legacy.py --model-path /work1/deming/shared/medusa-distributed-heads-Llama-3.1-405B_medusa_mlp_Llama-3.1-405B_medusa_5_lr_3e-05_layers_1 \
# --model-id meta-llama/Llama-3.1-405B

# cli file inference : 
accelerate launch --config_file /work1/deming/shared/.cache/huggingface/accelerate/default_config.yaml ./medusa/inference/cli.py  \
    --model /work1/deming/shared/medusa-distributed-heads-Llama-3.1-405B_medusa_mlp_Llama-3.1-405B_medusa_5_lr_3e-05_layers_1  \
    --base_model /work1/deming/shared/Llama-3.1-405B
# accelerate launch --config_file /work1/deming/shared/.cache/huggingface/accelerate/default_config.yaml ./medusa/inference/cli.py --model FasterDecoding/medusa-
# vicuna-7b-v1.3 --base_model lmsys/vicuna-7b-v1.3