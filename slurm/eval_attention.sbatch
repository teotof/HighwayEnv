#!/bin/bash
#SBATCH -J attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.out
#SBATCH -e /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.err

set -euo pipefail

module purge
module load anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate highwayfyp

cd /home/users/hadjitot/fyp/HighwayEnv

# Make sure old v2 models can import the extractor module
# If you have a root-level attention_extractor.py shim, $PWD is enough.
# If attention_extractor.py lives inside experiments/, include that too:
export PYTHONPATH=$PWD:$PWD/experiments

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${EXP_VERSION:=v2}"
: "${STEPS:=10000}"
: "${SCENARIO_NAME:=cf_v2}"
: "${SEED:=0}"
: "${ATTN_MODE:=}"

if [ -n "${ATTN_MODE}" ]; then
  : "${MODEL_PATH:=runs/models/ppo_attn_${ATTN_MODE}_${SCENARIO_NAME}_${EXP_VERSION}_seed${SEED}.zip}"
else
  : "${MODEL_PATH:=runs/models/ppo_attn_${SCENARIO_NAME}_${EXP_VERSION}_seed${SEED}.zip}"
fi

# IMPORTANT: export so Python sees them
export EXP_VERSION STEPS SCENARIO_NAME SEED ATTN_MODE MODEL_PATH

if [ -n "${ATTN_MODE}" ]; then
  OUT_DIR="runs/attn_logs/${EXP_VERSION}/${ATTN_MODE}/${SCENARIO_NAME}/seed${SEED}"
else
  OUT_DIR="runs/attn_logs/${EXP_VERSION}/${SCENARIO_NAME}/seed${SEED}"
fi
mkdir -p "$OUT_DIR"

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "SEED=$SEED"
echo "EXP_VERSION=$EXP_VERSION"
echo "ATTN_MODE=${ATTN_MODE:-legacy-name}"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"
echo "OUT_DIR=$OUT_DIR"

test -f "$MODEL_PATH" || { echo "Model not found: $MODEL_PATH"; exit 1; }

python experiments/eval_attention.py
python experiments/analyse_attention.py | tee "${OUT_DIR}/summary.txt"
