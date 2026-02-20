#!/bin/bash
#SBATCH -J base
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.out
#SBATCH -e /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.err

module purge
module load anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate highwayfyp

cd /home/users/hadjitot/fyp/HighwayEnv
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

: "${SCENARIO_NAME:=cf_v2}"
: "${EXP_VERSION:=v2}"
: "${TOTAL_TIMESTEPS:=500000}"
: "${N_ENVS:=4}"
: "${ATTN_MODE:=learned}"
: "${X_INDEX:=1}"
export SCENARIO_NAME EXP_VERSION TOTAL_TIMESTEPS N_ENVS ATTN_MODE X_INDEX

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "EXP_VERSION=$EXP_VERSION"
echo "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
echo "N_ENVS=$N_ENVS"
echo "ATTN_MODE=$ATTN_MODE"
echo "X_INDEX=$X_INDEX"

python experiments/train_ppo_attention.py

