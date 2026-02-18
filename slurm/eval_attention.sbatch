#!/bin/bash
#SBATCH -J attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16
#SBATCH -t 1-00:00:00
#SBATCH -o /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.out
#SBATCH -e /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.err

module purge
module load anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate highwayfyp

cd /home/users/hadjitot/fyp/HighwayEnv
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

: "${EXP_VERSION:=v2}"
: "${STEPS:=10000}"
: "${SCENARIO_NAME:=cf_v2}"
: "${SEED:=0}"

: "${MODEL_PATH:=runs/models/ppo_attn_${SCENARIO_NAME}_${EXP_VERSION}_seed${SEED}.zip}"

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "SEED=$SEED"
echo "EXP_VERSION=$EXP_VERSION"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"

python experiments/eval_attention.py
python experiments/analyse_attention.py | tee runs/attn_logs/${SCENARIO_NAME}/seed${SEED}/summary.txt

