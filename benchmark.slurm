#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=3:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
conda activate take_a_ride
python3 code/benchmark.py
"