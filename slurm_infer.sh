#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=training             # Name of the job
#SBATCH --output=output-%j.log            # Output log file
#SBATCH --error=error-%j.log              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=internal               # Partition to submit to
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
export PYTHONPATH=$PYTHONPATH:/work1/deming/seliny2/Medusa
ulimit -n 65536
# torchrun python3 -m medusa.inference.cli --model ../../shared/medusa-Llama-3.1-405B_medusa_mlp_Llama-3.1-405B_medusa_5_lr_3e-05_layers_1
accelerate launch --config_file /work1/deming/shared/.cache/huggingface/accelerate/default_config.yaml ./medusa/inference/cli.py \
    --model /work1/deming/shared/medusa-Llama-3.1-405B_medusa_mlp_Llama-3.1-405B_medusa_5_lr_3e-05_layers_1
