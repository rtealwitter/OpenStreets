#!/bin/bash

#SBATCH --job-name=qlearning
#SBATCH --open-mode=append
#SBATCH --output=./hpc_output/%x_%j.out
#SBATCH --error=./hpc_output/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rtealwitter@nyu.edu
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/OpenStreets/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate roads
python code/qlearning.py
"
