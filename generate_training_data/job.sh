#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name="hhc_gen"
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=y.clapper@vu.nl
cd
cd generate_training_data
LANG=en_US.UTF-8
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
pip install --force-reinstall numpy==1.24.3
pip install --force-reinstall scipy==1.10.1
python generate.py