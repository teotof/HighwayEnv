#!/bin/bash
#SBATCH -J shared_attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.out
#SBATCH -e /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.err

set -euo pipefail

module purge
module load anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate highwayfyp

cd /home/users/hadjitot/fyp/HighwayEnv
export PYTHONPATH=$PWD:$PWD/experiments
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${EXP_VERSION:=v10}"
: "${STEPS:=10000}"
: "${SCENARIO_NAME:=cf_v2}"
: "${SEED:=0}"
: "${ATTN_MODE:=learned}"
: "${CONTROLLED_VEHICLES:=2}"
: "${MODEL_DIR:=runs/models}"
: "${MODEL_PATH:=${MODEL_DIR}/ppo_shared_attn_${ATTN_MODE}_${SCENARIO_NAME}_${EXP_VERSION}_cv${CONTROLLED_VEHICLES}_seed${SEED}.zip}"

export EXP_VERSION STEPS SCENARIO_NAME SEED ATTN_MODE CONTROLLED_VEHICLES MODEL_DIR MODEL_PATH

OUT_DIR="runs/attn_logs_shared/${ATTN_MODE}/${SCENARIO_NAME}/cv${CONTROLLED_VEHICLES}/seed${SEED}"
mkdir -p "$OUT_DIR"

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "SEED=$SEED"
echo "EXP_VERSION=$EXP_VERSION"
echo "ATTN_MODE=$ATTN_MODE"
echo "CONTROLLED_VEHICLES=$CONTROLLED_VEHICLES"
echo "MODEL_DIR=$MODEL_DIR"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"
echo "OUT_DIR=$OUT_DIR"

test -f "$MODEL_PATH" || { echo "Model not found: $MODEL_PATH"; exit 1; }

python experiments/eval_shared_attention.py
python experiments/analyse_shared_attention.py | tee "${OUT_DIR}/summary.txt"
