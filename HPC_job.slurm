#!/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=python_test_output.txt
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=standard   # <-- change this if your cluster uses a different partition

module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2
python test_task.py
