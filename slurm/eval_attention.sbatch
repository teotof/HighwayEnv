#!/bin/bash
#SBATCH -J attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
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

: "${MODEL_PATH:=runs/models/ppo_attn_${SCENARIO_NAME}_${EXP_VERSION}_seed${SEED}.zip}"

# IMPORTANT: export so Python sees them
export EXP_VERSION STEPS SCENARIO_NAME SEED MODEL_PATH

OUT_DIR="runs/attn_logs/${SCENARIO_NAME}/seed${SEED}"
mkdir -p "$OUT_DIR"

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "SEED=$SEED"
echo "EXP_VERSION=$EXP_VERSION"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"
echo "OUT_DIR=$OUT_DIR"

test -f "$MODEL_PATH" || { echo "Model not found: $MODEL_PATH"; exit 1; }

python experiments/eval_attention.py
python experiments/analyse_attention.py | tee "${OUT_DIR}/summary.txt"