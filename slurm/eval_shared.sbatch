#!/bin/bash
#SBATCH -J shared_eval
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
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

: "${TRAIN_SCENARIO_NAME:=cf_v2}"
: "${EVAL_SCENARIO_NAME:=$TRAIN_SCENARIO_NAME}"
: "${EXP_VERSION:=v10}"
: "${SEEDS:=0,1,2}"
: "${CONTROLLED_VEHICLES:=2}"
: "${N_EVAL_EPISODES:=50}"
: "${MAX_STEPS:=0}"
: "${MODEL_DIR:=runs/models}"
export TRAIN_SCENARIO_NAME EVAL_SCENARIO_NAME EXP_VERSION SEEDS
export CONTROLLED_VEHICLES N_EVAL_EPISODES MAX_STEPS MODEL_DIR

echo "TRAIN_SCENARIO_NAME=$TRAIN_SCENARIO_NAME"
echo "EVAL_SCENARIO_NAME=$EVAL_SCENARIO_NAME"
echo "EXP_VERSION=$EXP_VERSION"
echo "SEEDS=$SEEDS"
echo "CONTROLLED_VEHICLES=$CONTROLLED_VEHICLES"
echo "N_EVAL_EPISODES=$N_EVAL_EPISODES"
echo "MAX_STEPS=$MAX_STEPS"
echo "MODEL_DIR=$MODEL_DIR"

python experiments/eval_shared_ppo.py
