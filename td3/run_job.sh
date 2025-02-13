#!/bin/bash --login
#SBATCH -J td3               # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani  # Which partition will run your job
#SBATCH --time=3-00:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2              # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=logs/job-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=logs/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=FAIL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=bela.umlauf@student.uni-tuebingen.de   # Email to which notifications will be sent


# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
#module load anaconda
conda activate rl_env

# Ensure environment activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment"
  exit 1
fi

# Change to the `td3` directory where the script is located
cd /mnt/qb/work/ludwig/lqb122/rl_project/td3

# Run your code
echo "-------- PYTHON OUTPUT ----------"
python3 train_agent.py --results_folder ./results/both/self_play --use_PER True --max_episodes 30000 --self_play_start 16000


# Check if Python script ran successfully
if [ $? -ne 0 ]; then
  echo "Python script failed"
  exit 1
fi

echo "---------------------------------"

# Deactivate environment again
conda deactivate

