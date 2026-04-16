#!/bin/bash
#SBATCH --export=NONE               
#SBATCH --job-name=pointcloud-classifier
#SBATCH --time=1-00:00:00           
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=24          
#SBATCH --mem=180G                  
#SBATCH --gres=gpu:a100:1           


module purge
module load GCCcore/12.2.0 Python/3.10.8 CUDA/11.3.1 

source venv/bin/activate

# Any print() statements in this script will now route to the log file above
python train.py --model transformer --epochs 1 --batch_size 64