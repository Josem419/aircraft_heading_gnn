#!/bin/bash
"""
Quick visualization runner - generates all basic visualizations.
Usage: ./scripts/quick_viz.sh [AIRPORT_ICAO]
"""

# Set defaults
AIRPORT_ICAO=${1:-KSEA}
PARQUET_PATH="data/processed/batches/states_2022-06-20-02.csv.parquet"
AIRPORTS_JSON="data/airports.json"

echo "Aircraft Heading GNN - Quick Visualization Suite"
echo "Airport: $AIRPORT_ICAO"
echo "Data: $PARQUET_PATH"
echo ""

# Source environment setup
source setup_env.sh

# Create output directories
mkdir -p plots analysis prediction_analysis

echo "1/3 - Generating trajectory plots..."
python3 scripts/visualize_trajectories.py \
    --parquet_path "$PARQUET_PATH" \
    --airports_json "$AIRPORTS_JSON" \
    --airport_icao "$AIRPORT_ICAO" \
    --max_trajs 1000 \
    --output_dir plots

echo "2/3 - Creating dataset analysis..."
python3 scripts/analyze_dataset.py \
    --parquet_path "$PARQUET_PATH" \
    --airports_json "$AIRPORTS_JSON" \
    --airport_icao "$AIRPORT_ICAO" \
    --num_snapshots 3 \
    --output_dir analysis

# echo "3/3 - Generating prediction analysis (demo)..."
python3 scripts/visualize_predictions.py \
    --model_path "checkpoints_parquet_first_run/best_model.pt" \
    --parquet_path "$PARQUET_PATH" \
    --airports_json "$AIRPORTS_JSON" \
    --airport_icao "$AIRPORT_ICAO" \
    --num_samples 50 \
    --output_dir prediction_analysis

echo ""
echo "Visualization complete!"
echo "Check these directories:"
echo "   - plots/ - Basic trajectory and heading plots"
echo "   - analysis/ - Dataset statistics and graph snapshots"  
echo "   - prediction_analysis/ - Model prediction analysis (demo)"