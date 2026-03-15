#!/bin/bash
#SBATCH -J shared_attn
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
: "${EXP_VERSION:=v10}"
: "${TOTAL_TIMESTEPS:=500000}"
: "${SEEDS:=0,1,2}"
: "${CONTROLLED_VEHICLES:=2}"
: "${N_WORLDS:=4}"
: "${VEC_MODE:=async}"
: "${TARGET_ROLLOUT_SIZE:=8192}"
: "${DEVICE:=cpu}"
: "${MODEL_DIR:=runs/models}"
: "${MONITOR_ROOT:=runs/monitor}"
: "${TB_DIR:=runs/tb}"
: "${ATTN_MODE:=learned}"
: "${X_INDEX:=1}"
: "${VX_INDEX:=3}"
: "${ORACLE_RULE:=leader_x}"
: "${TTC_EPS:=1e-6}"
export SCENARIO_NAME EXP_VERSION TOTAL_TIMESTEPS SEEDS CONTROLLED_VEHICLES
export N_WORLDS VEC_MODE TARGET_ROLLOUT_SIZE DEVICE MODEL_DIR MONITOR_ROOT TB_DIR
export ATTN_MODE X_INDEX VX_INDEX ORACLE_RULE TTC_EPS

echo "SCENARIO_NAME=$SCENARIO_NAME"
echo "EXP_VERSION=$EXP_VERSION"
echo "TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS"
echo "SEEDS=$SEEDS"
echo "CONTROLLED_VEHICLES=$CONTROLLED_VEHICLES"
echo "N_WORLDS=$N_WORLDS"
echo "VEC_MODE=$VEC_MODE"
echo "TARGET_ROLLOUT_SIZE=$TARGET_ROLLOUT_SIZE"
echo "DEVICE=$DEVICE"
echo "MODEL_DIR=$MODEL_DIR"
echo "MONITOR_ROOT=$MONITOR_ROOT"
echo "TB_DIR=$TB_DIR"
echo "ATTN_MODE=$ATTN_MODE"
echo "X_INDEX=$X_INDEX"
echo "VX_INDEX=$VX_INDEX"
echo "ORACLE_RULE=$ORACLE_RULE"
echo "TTC_EPS=$TTC_EPS"

python experiments/train_ppo_shared_attention.py
