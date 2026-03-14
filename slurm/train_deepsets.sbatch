#!/bin/bash
#SBATCH -J deepsets
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
: "${EXP_VERSION:=v9}"
: "${TOTAL_TIMESTEPS:=500000}"
: "${N_ENVS:=4}"
: "${DEEPSETS_HIDDEN:=64}"
export SCENARIO_NAME EXP_VERSION TOTAL_TIMESTEPS N_ENVS DEEPSETS_HIDDEN

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "EXP_VERSION=$EXP_VERSION"
echo "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
echo "N_ENVS=$N_ENVS"
echo "DEEPSETS_HIDDEN=$DEEPSETS_HIDDEN"

python experiments/train_ppo_deepsets.py
