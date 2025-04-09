import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.hybrid_lstm_gnn.data_utils import (
    load_data, preprocess_data_lstm, prepare_gnn_dataset, 
    save_processed_data, load_processed_data
)
from src.hybrid_lstm_gnn.lstm_model import (
    BallPredictorLSTM, train_lstm_model, evaluate_lstm_model, 
    visualize_predictions, create_demo_visualization
)
from src.hybrid_lstm_gnn.gnn_model import (
    BallPredictorGNN, TemporalGNN, train_gnn_model, 
    evaluate_gnn_model, visualize_gnn_graph
)
from src.hybrid_lstm_gnn.hybrid_model import (
    HybridLSTMGNN, train_hybrid_model, evaluate_hybrid_model,
    visualize_hybrid_predictions, apply_player_proximity
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Hybrid LSTM-GNN Model for Ball Coordinate Prediction')
    
    parser.add_argument('--match', type=int, default=0, help='Match number to use (default: 0)')
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'test', 'demo'],
                        help='Mode: train, test, or demo (default: demo)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for models (default: 5)')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use (default: 1000)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for outputs (default: output)')
    
    return parser.parse_args()

def run_demo(match_num, sequence_length=5, sample_size=1000, output_dir='output'):
    """Run a demonstration of the hybrid model with a small sample of data."""
    print(f"Running demonstration with match {match_num}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    ball_df, home_df, away_df = load_data(match_num)
    
    # Use a subset of data for the demo
    if len(ball_df) > sample_size:
        start_idx = 1000  # Skip the first 1000 frames to avoid potential initialization issues
        ball_df = ball_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        home_df = home_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        away_df = away_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        print(f"Using {sample_size} samples starting from frame {start_idx}")
    
    # Create a demonstration visualization
    print("Creating player influence demonstration...")
    create_demo_visualization(proximity_threshold=15.0)
    
    # Process data for LSTM
    print("Processing data for LSTM...")
    try:
        lstm_X, lstm_y, lstm_scaler = preprocess_data_lstm(
            home_df, away_df, ball_df, 
            n_steps=sequence_length, n_future=1
        )
        print(f"LSTM data processed: X shape {lstm_X.shape}, y shape {lstm_y.shape}")
    except Exception as e:
        print(f"Error processing LSTM data: {e}")
        return
    
    # Process data for GNN
    print("Processing data for GNN...")
    try:
        gnn_sequences, gnn_targets = prepare_gnn_dataset(
            home_df, away_df, ball_df,
            sequence_length=sequence_length, prediction_steps=1
        )
        print(f"GNN data processed: {len(gnn_sequences)} sequences")
    except Exception as e:
        print(f"Error processing GNN data: {e}")
        return
    
    # Create simple models for demonstration
    print("Creating models...")
    
    # LSTM model
    input_size = lstm_X.shape[2]
    lstm_model = BallPredictorLSTM(input_size, hidden_size=64)
    
    # GNN model
    if gnn_sequences and len(gnn_sequences) > 0 and gnn_sequences[0] and len(gnn_sequences[0]) > 0:
        input_dim = gnn_sequences[0][0].x.shape[1]
        gnn_model = TemporalGNN(input_dim, hidden_dim=32, lstm_hidden_dim=64)
    else:
        print("Error: GNN sequences are empty or invalid")
        return
    
    # Hybrid model
    hybrid_model = HybridLSTMGNN(input_size, input_dim, lstm_hidden_size=64, gnn_hidden_dim=32)
    
    # Make predictions with untrained models (just for demonstration)
    print("Making predictions with untrained models (for demonstration only)...")
    
    # Convert to PyTorch tensors
    lstm_tensor = torch.tensor(lstm_X[:10], dtype=torch.float32)
    
    # LSTM predictions
    lstm_model.eval()
    with torch.no_grad():
        lstm_pred = lstm_model(lstm_tensor).numpy()
    
    # GNN predictions
    gnn_model.eval()
    with torch.no_grad():
        gnn_pred = []
        for i in range(min(10, len(gnn_sequences))):
            pred = gnn_model(gnn_sequences[i]).numpy()
            gnn_pred.append(pred[0])
        gnn_pred = np.array(gnn_pred)
    
    # Hybrid predictions
    hybrid_model.eval()
    with torch.no_grad():
        hybrid_pred = []
        for i in range(min(10, len(gnn_sequences))):
            pred = hybrid_model(lstm_tensor[i:i+1], gnn_sequences[i]).numpy()
            hybrid_pred.append(pred[0])
        hybrid_pred = np.array(hybrid_pred)
    
    # Visualize predictions
    print("Visualizing predictions...")
    
    # Use a small segment for visualization
    vis_start = 0
    vis_end = min(10, len(lstm_y))
    
    # Visualize LSTM predictions
    visualize_predictions(
        lstm_y[vis_start:vis_end], lstm_pred, 
        home_df, away_df, vis_start, vis_end
    )
    
    # Visualize GNN graph structure
    if gnn_sequences and len(gnn_sequences) > 0 and gnn_sequences[0] and len(gnn_sequences[0]) > 0:
        visualize_gnn_graph(gnn_sequences[0][-1], "Sample Graph Structure")
    
    # Visualize hybrid predictions
    visualize_hybrid_predictions(
        lstm_y[vis_start:vis_end], lstm_pred, gnn_pred, hybrid_pred,
        home_df, away_df, vis_start, vis_end
    )
    
    print("\nDemonstration completed successfully!")
    print(f"Visualizations saved to current directory.")
    print("Note: The models are untrained, so predictions are random.")
    print("To train the models, use --mode train with more epochs.")

def main():
    """Main function to run the hybrid model."""
    args = parse_args()
    
    if args.mode == 'demo':
        run_demo(args.match, args.sequence_length, args.sample_size, args.output_dir)
    elif args.mode == 'train':
        print("Training mode not implemented in this script. Please use main.py from the hybrid_lstm_gnn package.")
    elif args.mode == 'test':
        print("Test mode not implemented in this script. Please use main.py from the hybrid_lstm_gnn package.")

if __name__ == "__main__":
    main()
