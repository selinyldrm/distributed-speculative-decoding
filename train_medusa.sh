#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=train-8b         # Name of the job
#SBATCH --output=output-%j.log            # Output log file
#SBATCH --error=error-%j.log              # Error log file
#SBATCH --time=7:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi3008x_long          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda deactivate
conda activate medusa-head-training
cd /work1/deming/seliny2/distributed-speculative-decoding
export HOME=/work1/deming/shared
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:/lib
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/work1/deming/seliny2/distributed-speculative-decoding/
ulimit -n 65536
pip3 install -r requirements.txt

torchrun --nproc_per_node=8 ./medusa/train/train_legacy.py \
 --model_name_or_path /work1/deming/shared/Llama-3.1-405B    \
 --output_dir /work1/deming/shared/medusa_fixed-405B-5heads \
 --deepspeed /work1/deming/seliny2/axolotl/deepspeed/zero3-offload.json \
 --fp16 True \
 --per_device_train_batch_size 4 \
 --gradient_accumulation_steps 1 
