#!/bin/bash

# Exit on error
set -e

# Create a virtual environment (if not already created)
if [ ! -d "env" ]; then
  echo "Creating virtual environment..."
  python3 -m venv env
else
  echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete. Virtual environment is ready."
