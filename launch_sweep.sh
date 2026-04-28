#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=Multihead_new_features

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=/home/ethan2/logs/Multihead_conc_%A_%a.out
#SBATCH --error=/home/ethan2/logs/Multihead_conc_%A_%a.err
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=1000000000:00:00

## How many CPUs to request. Maximum is 124.
#SBATCH --cpus-per-task=20

## How much memory to request in MB. Maximum is 460GB.
#SBATCH --mem=16000

#SBATCH --gres=mps:20
#SBATCH --array=0-4

## You can also request a percentage of one GPU.
## Example to get 20% of a GPU.
## This approach has severe limitation as it can only be used by
## a single user at a time. More tests should be performed whether this
## ok to use it. Please experiment and report findings.

set -e

#cd /home/ethan2/GrowthNet/experiments/MultiHead/
# The below env variables can eventually help setting up your workload.

echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# Sanitize the inherited environment: sbatch propagates the submitting shell's
# environment, which may have a virtualenv (e.g. boltz_docking) activated.
# Strip any active venv from PATH and clear VIRTUAL_ENV before activating the uv venv.
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT

source /home/ethan2/venvs/GrowthCurve/bin/activate

# Use "python -m wandb agent" so that wandb spawns sweep runs via sys.executable
# (the conda env's Python 3.10) rather than whatever "python" resolves to in PATH.
#python -u accelerate launch --num_processes 3 sweep_multihead.py
python -m wandb agent 'ethan_personal/Predictive model final sweep/vxc78kac'
# A dummy and useless `sleep` to give you time to see your job with `squeue`.
sleep 20s