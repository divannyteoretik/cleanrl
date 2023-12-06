#!/bin/bash
#SBATCH -t960                                           # 960min=16hours limit
#SBATCH -oReport-%j.out                                 # Combined output and error messages file

cd $SLURM_SUBMIT_DIR                                    # Change to working directory

# Not sure we need this if we have miniconda
module load anaconda3                           # Load module dependencies

N_RESET_STEPS=${N_RESET_STEPS:-1000001} # no resets
SEED=${SEED:-"1"}
ENV=${ENV:-Humanoid-v4}
N_STEPS=${N_STEPS:-1000000}
N_UPDATES_PER_STEP=${N_UPDATES_PER_STEP:-1}

DO_RESETS=`echo "$N_STEPS > $N_RESET_STEPS" | bc`

echo "initializing conda"
source ~/scratch/miniconda3/etc/profile.d/conda.sh

echo "activating conda env"
conda activate py38_pip
TIMESTAMP=`date +%s`
LOG_NAME="${ENV}__total_${N_STEPS}__resets${DO_RESETS}__s_${SEED}_${TIMESTAMP}.log"
echo $LOG_NAME

nvidia-smi

echo "launching script"
python cleanrl/sac_continuous_action_resets.py --env-id "$ENV" --seed $SEED --n-reset-steps $N_RESET_STEPS --total_timesteps $N_STEPS --n-updates-per-step $N_UPDATES_PER_STEP | tee $LOG_NAME

echo "DONE"
