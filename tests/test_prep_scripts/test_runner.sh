#!/bin/bash
#
#SBATCH --job-name=test-runner
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30

# Load IDL
module load idl/8.4

srun idl -e "histogram_runner"

srun idl -e "rebin_runner"
