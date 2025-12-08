#!/bin/bash
# Aircraft Heading GNN Development Environment Setup
# Source this file to set up the Python path for development
# Usage: source setup_env.sh

# Get the directory where this script is located (repo root)
export AIRCRAFT_GNN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add src directory to Python path
export PYTHONPATH="${AIRCRAFT_GNN_ROOT}/src:${PYTHONPATH}"

# Activate the virtual environment if it exists
if [ -d "${AIRCRAFT_GNN_ROOT}/.gnn_env" ]; then
    source "${AIRCRAFT_GNN_ROOT}/.gnn_env/bin/activate"
    echo "Activated .gnn_env virtual environment"
else
    echo "Virtual environment .gnn_env not found"
fi

echo "Added ${AIRCRAFT_GNN_ROOT}/src to PYTHONPATH"
echo "Environment setup complete"