#!/bin/bash
#!/bin/bash
#SBATCH --ntasks=1             # Number of tasks to run (equal to 1 CPU/core per task)
#SBATCH --gres=gpu:1           # Number of GPUs (per node)
#SBATCH --mem=4000M            # Memory (per node)
#SBATCH --time=0-03:00         # Time (DD-HH:MM)
#SBATCH -o load.out        # File to which STDOUT will be written, %j inserts job ID
#SBATCH -e load.err        # File to which STDERR will be written, %j inserts job ID

set -e

# Create a virtual environment (if not already created)
if [ ! -d "env" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete. Virtual environment is ready."
