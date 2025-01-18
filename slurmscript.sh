#!/bin/bash
#SBATCH --ntasks=1             # Number of tasks to run (equal to 1 CPU/core per task)
#SBATCH --gres=gpu:1           # Number of GPUs (per node)
#SBATCH --mem=4000M            # Memory (per node)
#SBATCH --time=0-03:00         # Time (DD-HH:MM)
#SBATCH -o myoutput.out        # File to which STDOUT will be written, %j inserts job ID
#SBATCH -e myerrors.err        # File to which STDERR will be written, %j inserts job ID

pwd                            # Print the working directory
source .venv/bin/activate      # Activate virtual environment
python testcode.py             # Run Python script
