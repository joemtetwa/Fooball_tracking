import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import time

from data_utils import (
    load_data, preprocess_data_lstm, prepare_gnn_dataset, 
    save_processed_data, load_processed_data
)
from lstm_model import (
    train_lstm_model, evaluate_lstm_model, predict_with_lstm, 
    plot_training_history, visualize_predictions, create_demo_visualization,
    apply_player_proximity
)
from gnn_model import (
    train_gnn_model, evaluate_gnn_model, predict_with_gnn,
    plot_gnn_training_history, visualize_gnn_graph, visualize_gnn_predictions
)
from hybrid_model import (
    train_hybrid_model, evaluate_hybrid_model, predict_with_hybrid,
    plot_hybrid_training_history, visualize_hybrid_predictions,
    compare_model_performance, plot_performance_comparison
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hybrid LSTM-GNN Model for Ball Coordinate Prediction')
    
    parser.add_argument('--match', type=int, default=0, help='Match number to use for training/testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='Mode: train, test, or demo')
    parser.add_argument('--model', type=str, default='hybrid', choices=['lstm', 'gnn', 'hybrid'],
                        help='Model type to train/test')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for LSTM/GNN')
    parser.add_argument('--prediction_steps', type=int, default=1, help='Number of steps to predict')
    parser.add_argument('--proximity_threshold', type=float, default=15.0, 
                        help='Threshold distance for player influence (meters)')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data',
                        help='Directory to save/load processed data')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save/load models')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--demo', action='store_true', help='Run demonstration visualization')
    
    return parser.parse_args()

def main():
    """Main function to run the hybrid model."""
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set model paths
    lstm_model_path = os.path.join(args.model_dir, f'lstm_model_match_{args.match}.pt')
    gnn_model_path = os.path.join(args.model_dir, f'gnn_model_match_{args.match}.pt')
    hybrid_model_path = os.path.join(args.model_dir, f'hybrid_model_match_{args.match}.pt')
    
    # Run demonstration if requested
    if args.demo:
        print("Running demonstration visualization...")
        create_demo_visualization(proximity_threshold=args.proximity_threshold)
        print("Demonstration visualization saved to player_influence_demo.png")
        return
    
    # Load data
    print(f"Loading data for match {args.match}...")
    ball_df, home_df, away_df = load_data(args.match)
    
    # Process data based on model type
    if args.mode == 'train':
        print("Processing data for training...")
        
        # LSTM data preprocessing
        if args.model in ['lstm', 'hybrid']:
            print("Preprocessing data for LSTM...")
            lstm_X, lstm_y, lstm_scaler = preprocess_data_lstm(
                home_df, away_df, ball_df, 
                n_steps=args.sequence_length, n_future=args.prediction_steps
            )
            
            # Save processed LSTM data
            save_processed_data(
                lstm_X, lstm_y, lstm_scaler,
                args.processed_data_dir, prefix="lstm"
            )
        
        # GNN data preprocessing
        if args.model in ['gnn', 'hybrid']:
            print("Preparing data for GNN...")
            gnn_sequences, gnn_targets = prepare_gnn_dataset(
                home_df, away_df, ball_df,
                sequence_length=args.sequence_length, prediction_steps=args.prediction_steps
            )
            
            # Save processed GNN data (only targets, sequences are objects)
            save_processed_data(
                np.zeros((1, 1)), gnn_targets, None,
                args.processed_data_dir, prefix="gnn"
            )
            print(f"GNN sequences prepared: {len(gnn_sequences)}")
        
        # Split data for training
        if args.model == 'lstm':
            # Split LSTM data
            lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test = train_test_split(
                lstm_X, lstm_y, test_size=0.2, random_state=42
            )
            
            # Train LSTM model
            print("Training LSTM model...")
            lstm_model, lstm_history = train_lstm_model(
                lstm_X_train, lstm_y_train,
                lstm_X_test, lstm_y_test,
                lstm_X.shape[2],
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_save_path=lstm_model_path
            )
            
            # Plot training history
            plot_training_history(lstm_history)
            
        elif args.model == 'gnn':
            # Train GNN model
            print("Training GNN model...")
            gnn_model, gnn_history = train_gnn_model(
                gnn_sequences,
                gnn_targets,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_save_path=gnn_model_path
            )
            
            # Plot training history
            plot_gnn_training_history(gnn_history)
            
        elif args.model == 'hybrid':
            # Train hybrid model
            print("Training hybrid model...")
            hybrid_model, hybrid_history = train_hybrid_model(
                lstm_X,
                gnn_sequences,
                lstm_y,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                model_save_path=hybrid_model_path
            )
            
            # Plot training history
            plot_hybrid_training_history(hybrid_history)
    
    elif args.mode == 'test':
        print("Loading processed data for testing...")
        
        # Load LSTM data
        if args.model in ['lstm', 'hybrid']:
            lstm_X, lstm_y, lstm_scaler = load_processed_data(
                args.processed_data_dir, prefix="lstm"
            )
        
        # Load GNN data
        if args.model in ['gnn', 'hybrid']:
            _, gnn_targets, _ = load_processed_data(
                args.processed_data_dir, prefix="gnn"
            )
            
            # Recreate GNN sequences (can't save PyTorch Geometric objects easily)
            print("Recreating GNN sequences...")
            gnn_sequences, _ = prepare_gnn_dataset(
                home_df, away_df, ball_df,
                sequence_length=args.sequence_length, prediction_steps=args.prediction_steps
            )
        
        # Test models
        if args.model == 'lstm':
            # Load LSTM model
            print("Loading LSTM model...")
            lstm_model = torch.load(lstm_model_path)
            
            # Evaluate LSTM model
            print("Evaluating LSTM model...")
            lstm_loss, lstm_predictions = evaluate_lstm_model(
                lstm_model, lstm_X, lstm_y, batch_size=args.batch_size
            )
            
            # Visualize predictions
            if args.visualize:
                print("Visualizing LSTM predictions...")
                # Select a segment for visualization
                start_idx = 100
                end_idx = 150
                
                visualize_predictions(
                    lstm_y, lstm_predictions, home_df, away_df,
                    start_idx, end_idx
                )
                
        elif args.model == 'gnn':
            # Load GNN model
            print("Loading GNN model...")
            gnn_model = torch.load(gnn_model_path)
            
            # Evaluate GNN model
            print("Evaluating GNN model...")
            gnn_loss, gnn_predictions = evaluate_gnn_model(
                gnn_model, gnn_sequences, gnn_targets
            )
            
            # Visualize predictions
            if args.visualize:
                print("Visualizing GNN predictions...")
                # Select a segment for visualization
                start_idx = 100
                end_idx = 150
                
                visualize_gnn_predictions(
                    gnn_targets, gnn_predictions, gnn_sequences,
                    start_idx, end_idx
                )
                
                # Visualize a single graph
                visualize_gnn_graph(gnn_sequences[end_idx-1][-1], "Graph Structure at Frame")
                
        elif args.model == 'hybrid':
            # Load all models for comparison
            print("Loading models for comparison...")
            
            # Load LSTM model and make predictions
            lstm_model = torch.load(lstm_model_path)
            _, lstm_predictions = evaluate_lstm_model(
                lstm_model, lstm_X, lstm_y, batch_size=args.batch_size
            )
            
            # Load GNN model and make predictions
            gnn_model = torch.load(gnn_model_path)
            _, gnn_predictions = evaluate_gnn_model(
                gnn_model, gnn_sequences, gnn_targets
            )
            
            # Load hybrid model and make predictions
            hybrid_model = torch.load(hybrid_model_path)
            hybrid_loss, hybrid_predictions = evaluate_hybrid_model(
                hybrid_model, lstm_X, gnn_sequences, lstm_y
            )
            
            # Compare model performance
            metrics = compare_model_performance(
                lstm_y, lstm_predictions, gnn_predictions, hybrid_predictions
            )
            
            # Plot performance comparison
            plot_performance_comparison(metrics)
            
            # Visualize predictions
            if args.visualize:
                print("Visualizing hybrid model predictions...")
                # Select a segment for visualization
                start_idx = 100
                end_idx = 150
                
                visualize_hybrid_predictions(
                    lstm_y, lstm_predictions, gnn_predictions, hybrid_predictions,
                    home_df, away_df, start_idx, end_idx
                )
                
                # Apply player proximity to hybrid predictions
                print("\nApplying player proximity to hybrid predictions...")
                adjusted_predictions = []
                
                for i in range(len(hybrid_predictions)):
                    adjusted_pos = apply_player_proximity(
                        hybrid_predictions[i], home_df, away_df, i,
                        proximity_threshold=args.proximity_threshold
                    )
                    adjusted_predictions.append(adjusted_pos)
                
                adjusted_predictions = np.array(adjusted_predictions)
                
                # Compare performance before and after player proximity adjustment
                print("\nPerformance after player proximity adjustment:")
                adjusted_metrics = compare_model_performance(
                    lstm_y, lstm_predictions, gnn_predictions, adjusted_predictions
                )
                
                # Visualize adjusted predictions
                visualize_hybrid_predictions(
                    lstm_y, lstm_predictions, hybrid_predictions, adjusted_predictions,
                    home_df, away_df, start_idx, end_idx
                )
                print("Visualization saved to hybrid_prediction_comparison.png")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
