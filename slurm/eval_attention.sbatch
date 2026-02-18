#!/bin/bash
#SBATCH -J attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
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
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${EXP_VERSION:=v2}"
: "${STEPS:=10000}"
: "${SCENARIO_NAME:=cf_v2}"
: "${SEED:=0}"

: "${MODEL_PATH:=runs/models/ppo_attn_${SCENARIO_NAME}_${EXP_VERSION}_seed${SEED}.zip}"

OUT_DIR="runs/attn_logs/${SCENARIO_NAME}/seed${SEED}"
mkdir -p "$OUT_DIR"

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "SEED=$SEED"
echo "EXP_VERSION=$EXP_VERSION"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"
echo "OUT_DIR=$OUT_DIR"

python experiments/eval_attention.py
python experiments/analyse_attention.py | tee "${OUT_DIR}/summary.txt"

