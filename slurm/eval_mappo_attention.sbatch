#!/bin/bash
#SBATCH -J mappo_attnlog
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0-12:00:00
#SBATCH -o /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.out
#SBATCH -e /home/users/hadjitot/fyp/HighwayEnv/slurm_out/%x_%j.err

module purge
module load anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate highwayfyp

cd /home/users/hadjitot/fyp/HighwayEnv
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

: "${SCENARIO_NAME:=cf_obs15}"
: "${EXP_VERSION:=v11_attn}"
: "${SEED:=0}"
: "${ATTN_MODE:=learned}"
: "${CONTROLLED_VEHICLES:=2}"
: "${MODEL_DIR:=runs/models}"
: "${MODEL_PATH:=}"
: "${STEPS:=5000}"
: "${DEVICE:=cpu}"
export SCENARIO_NAME EXP_VERSION SEED ATTN_MODE CONTROLLED_VEHICLES
export MODEL_DIR STEPS DEVICE
if [[ -n "${MODEL_PATH}" ]]; then
  export MODEL_PATH
fi

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "EXP_VERSION=$EXP_VERSION"
echo "SEED=$SEED"
echo "ATTN_MODE=$ATTN_MODE"
echo "CONTROLLED_VEHICLES=$CONTROLLED_VEHICLES"
echo "MODEL_DIR=$MODEL_DIR"
echo "MODEL_PATH=$MODEL_PATH"
echo "STEPS=$STEPS"
echo "DEVICE=$DEVICE"

python experiments/eval_mappo_attention.py
