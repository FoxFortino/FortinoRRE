#!/bin/bash -l

#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G
# SBATCH --gpus=tesla_t4:1 --ntasks=1 --mem=32G

#SBATCH --array=6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,55,60,65,70,75,80,85,90,95,100,150,200,300,400,500,600,700,738

#SBATCH --job-name=dash
#SBATCH --partition=idle
#SBATCH --time=4-00:00:00
#SBATCH --mail-user="fortino@udel.edu"
#SBATCH --mail-type=ALL

#SBATCH --requeue

#SBATCH --export=ALL

UD_QUIET_JOB_SETUP=YES
# vpkg_require anaconda/5.3.1:python3
. /opt/shared/slurm/templates/libexec/common.sh

bash /home/2649/.bashrc

# conda activate fox
conda info
python /home/2649/repos/FortinoRRE/scripts/train.py $SLURM_ARRAY_TASK_ID
