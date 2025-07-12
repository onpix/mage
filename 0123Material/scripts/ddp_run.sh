#!/bin/bash
#SBATCH --requeue
#SBATCH --quotatype=spot
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --get-user-env
#SBATCH --chdir=/mnt/petrelfs/wangzhenwei/projects/ThemeStationPro/InstantGTA3
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -x -e
ulimit -c 0

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SRUN_MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
SRUN_MASTER_PORT=$((RANDOM % 1011 + 13511))

export GPUS_PER_NODE=8

srun accelerate launch --config_file ./acc_configs/gpu8.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --rdzv_conf rdzv_id=$SLURM_JOB_ID \
    --main_process_ip $SRUN_MASTER_ADDR \
    --main_process_port $SRUN_MASTER_PORT \
    "$@"