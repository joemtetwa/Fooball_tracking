import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import (
    load_data, preprocess_data_lstm, prepare_gnn_dataset
)
from src.hybrid_lstm_gnn.lstm_model import (
    train_lstm_model, evaluate_lstm_model, predict_with_lstm
)
from src.hybrid_lstm_gnn.gnn_model import (
    train_gnn_model, evaluate_gnn_model, predict_with_gnn
)
from src.hybrid_lstm_gnn.hybrid_model import (
    train_hybrid_model, evaluate_hybrid_model, predict_with_hybrid
)

# Import pass analysis and enhanced model components
from pass_analysis import detect_passes, calculate_pass_probabilities
from src.hybrid_lstm_gnn.pass_analysis_integration import (
    create_pass_probability_matrix, extract_pass_features_for_frame, create_enhanced_graph
)
from src.hybrid_lstm_gnn.enhanced_hybrid_model import (
    EnhancedHybridModel, train_enhanced_hybrid_model, evaluate_enhanced_hybrid_model,
    plot_enhanced_training_history, visualize_enhanced_predictions, compare_enhanced_model_performance
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Hybrid LSTM-GNN Model with Pass Analysis')
    
    parser.add_argument('--train_matches', type=str, default='0,1,2', 
                        help='Comma-separated list of match numbers to use for training')
    parser.add_argument('--val_match', type=int, default=3, 
                        help='Match number to use for validation')
    parser.add_argument('--test_match', type=int, default=4, 
                        help='Match number to use for testing/prediction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo', 'predict'],
                        help='Mode: train, test, demo, or predict')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length for LSTM/GNN')
    parser.add_argument('--prediction_steps', type=int, default=1, help='Number of steps to predict')
    parser.add_argument('--proximity_threshold', type=float, default=5.0, 
                        help='Threshold distance for proximity edges (meters)')
    parser.add_argument('--pass_prob_threshold', type=float, default=0.1, 
                        help='Threshold for pass probability edges')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save/load models')
    parser.add_argument('--output_dir', type=str, default='enhanced_output',
                        help='Directory to save output files')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--all_frames', action='store_true', 
                        help='Process all frames instead of a subset')
    
    return parser.parse_args()


def prepare_pass_features(passes, home_df, away_df, ball_df, 
                         home_pass_matrix, away_pass_matrix, start_idx, num_samples):
    """
    Prepare pass features for all samples.
    
    Args:
        passes: List of pass tuples
        home_df, away_df: DataFrames with player positions
        ball_df: DataFrame with ball position
        home_pass_matrix, away_pass_matrix: Pass probability matrices
        start_idx: Starting frame index
        num_samples: Number of samples to process
        
    Returns:
        Array of pass features
    """
    pass_features = []
    
    # Handle empty pass matrices (prediction mode)
    if home_pass_matrix.empty or away_pass_matrix.empty:
        print("Using default pass features for prediction mode")
        # Create default features (all zeros) for prediction mode
        default_features = np.zeros(10)  # Adjust size based on your feature extraction
        return np.array([default_features] * num_samples)
    
    for i in range(num_samples):
        # Calculate frame index
        frame_idx = start_idx + i
        
        # Extract features
        features = extract_pass_features_for_frame(
            passes, home_df, away_df, ball_df, frame_idx,
            home_pass_matrix, away_pass_matrix,
            prediction_mode=(home_pass_matrix.empty or away_pass_matrix.empty)
        )
        
        # Check if features is already a numpy array or if it's a torch tensor
        if isinstance(features, torch.Tensor):
            pass_features.append(features.numpy())
        else:
            pass_features.append(features)
    
    return np.array(pass_features)


def prepare_enhanced_gnn_dataset(home_df, away_df, ball_df, 
                               home_pass_matrix, away_pass_matrix,
                               sequence_length=5, prediction_steps=1,
                               proximity_threshold=5.0, pass_prob_threshold=0.1):
    """
    Prepare enhanced GNN dataset with pass probability edges.
    
    Args:
        home_df, away_df: DataFrames with player positions
        ball_df: DataFrame with ball position
        home_pass_matrix, away_pass_matrix: Pass probability matrices
        sequence_length: Length of graph sequences
        prediction_steps: Number of steps to predict
        proximity_threshold: Threshold for proximity edges
        pass_prob_threshold: Threshold for pass probability edges
        
    Returns:
        List of graph sequences and target ball positions
    """
    # Create list to store graph sequences and targets
    graph_sequences = []
    targets = []
    
    for t in range(sequence_length, len(ball_df) - prediction_steps):
        # Create sequence of graphs
        sequence = []
        valid_sequence = True
        
        for i in range(t - sequence_length, t):
            # Create enhanced graph for this frame
            graph = create_enhanced_graph(
                home_df, away_df, ball_df, i,
                home_pass_matrix, away_pass_matrix,
                proximity_threshold, pass_prob_threshold
            )
            
            # Skip if graph creation failed
            if graph is None:
                valid_sequence = False
                break
                
            sequence.append(graph)
        
        # Skip if any graph in the sequence is invalid
        if not valid_sequence or len(sequence) != sequence_length:
            continue
            
        # Add sequence to dataset
        graph_sequences.append(sequence)
        
        # Target is the future ball position
        future_idx = t + prediction_steps
        if future_idx < len(ball_df):
            target = ball_df.iloc[future_idx][['ball_x', 'ball_y']].values
            targets.append(target)
    
    return graph_sequences, np.array(targets)


def main():
    """Main function to run the enhanced hybrid model."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse training matches
    train_match_nums = [int(m) for m in args.train_matches.split(',')]
    
    # Set model paths
    model_suffix = f"train_{args.train_matches}_val_{args.val_match}"
    lstm_model_path = os.path.join(args.model_dir, f"lstm_model_{model_suffix}.pt")
    gnn_model_path = os.path.join(args.model_dir, f"gnn_model_{model_suffix}.pt")
    enhanced_model_path = os.path.join(args.model_dir, f"enhanced_model_{model_suffix}.pt")
    
    # Function to load and process data from multiple matches
    def load_and_process_matches(match_nums, mode='train'):
        all_ball_dfs = []
        all_home_dfs = []
        all_away_dfs = []
        all_passes = []
        all_home_pass_matrices = []
        all_away_pass_matrices = []
        
        for match_num in match_nums:
            print(f"Loading data for match {match_num}...")
            try:
                # Use prediction_mode=True for test and predict modes
                prediction_mode = mode in ['test', 'predict']
                ball_df, home_df, away_df = load_data(match_num, prediction_mode=prediction_mode)
                
                # Optionally use a subset of data for faster processing
                if not args.all_frames and mode != 'predict':
                    # Use a reasonable subset size based on mode
                    if mode == 'train':
                        subset_size = 5000  # Larger for training
                    elif mode == 'val':
                        subset_size = 2000  # Medium for validation
                    else:  # test or demo
                        subset_size = 1000  # Smaller for testing/demo
                    
                    start_idx = 1000  # Skip initial frames which might have setup issues
                    ball_df = ball_df.iloc[start_idx:start_idx+subset_size].reset_index(drop=True)
                    home_df = home_df.iloc[start_idx:start_idx+subset_size].reset_index(drop=True)
                    away_df = away_df.iloc[start_idx:start_idx+subset_size].reset_index(drop=True)
                    print(f"Using subset of {subset_size} frames starting at index {start_idx}")
                else:
                    print(f"Using all {len(ball_df)} frames")
                
                # Detect passes
                print("Detecting passes...")
                # Use prediction_mode parameter for detect_passes
                passes = detect_passes(ball_df, home_df, away_df, 
                                      possession_radius=2.0, ball_speed_threshold=5.0,
                                      prediction_mode=(mode in ['test', 'predict']))
                
                # Calculate pass probabilities
                print("Calculating pass probabilities...")
                home_pass_matrix = create_pass_probability_matrix(passes, 'home_')
                away_pass_matrix = create_pass_probability_matrix(passes, 'away_')
                
                all_ball_dfs.append(ball_df)
                all_home_dfs.append(home_df)
                all_away_dfs.append(away_df)
                all_passes.append(passes)
                all_home_pass_matrices.append(home_pass_matrix)
                all_away_pass_matrices.append(away_pass_matrix)
            except Exception as e:
                print(f"Error loading match {match_num}: {e}")
        
        return all_ball_dfs, all_home_dfs, all_away_dfs, all_passes, all_home_pass_matrices, all_away_pass_matrices
    
    # Load data based on mode
    if args.mode == 'train':
        # Load training and validation data
        train_data = load_and_process_matches(train_match_nums, 'train')
        val_data = load_and_process_matches([args.val_match], 'val')
        
        train_ball_dfs, train_home_dfs, train_away_dfs, train_passes, train_home_matrices, train_away_matrices = train_data
        val_ball_dfs, val_home_dfs, val_away_dfs, val_passes, val_home_matrices, val_away_matrices = val_data
    elif args.mode == 'test' or args.mode == 'predict':
        # Load test data
        test_data = load_and_process_matches([args.test_match], args.mode)
        test_ball_dfs, test_home_dfs, test_away_dfs, test_passes, test_home_matrices, test_away_matrices = test_data
    else:  # demo mode
        # Load demo data (using match 0 for simplicity)
        demo_data = load_and_process_matches([0], 'demo')
        demo_ball_dfs, demo_home_dfs, demo_away_dfs, demo_passes, demo_home_matrices, demo_away_matrices = demo_data
        
        # Use the first match for demo
        ball_df, home_df, away_df = demo_ball_dfs[0], demo_home_dfs[0], demo_away_dfs[0]
        passes = demo_passes[0]
        home_pass_matrix = demo_home_matrices[0]
        away_pass_matrix = demo_away_matrices[0]
    
    if args.mode == 'demo':
        # Detect passes (already done in load_and_process_matches for other modes)
        print("Detecting passes...")
        passes = detect_passes(
            ball_df, home_df, away_df, 
            possession_radius=2.0, 
            ball_speed_threshold=5.0
        )
    
    if args.mode == 'demo':
        # Calculate pass probabilities (already done in load_and_process_matches for other modes)
        print("Calculating pass probabilities...")
        home_pass_matrix = create_pass_probability_matrix(passes, 'home_')
        away_pass_matrix = create_pass_probability_matrix(passes, 'away_')
    
    # Save pass matrices (only in demo mode)
    if args.mode == 'demo':
        home_pass_matrix.to_csv(os.path.join(args.output_dir, 'home_pass_matrix_demo.csv'))
        away_pass_matrix.to_csv(os.path.join(args.output_dir, 'away_pass_matrix_demo.csv'))
    
    if args.mode == 'train':
        # Lists to store processed data from all training matches
        all_lstm_X = []
        all_lstm_y = []
        all_pass_features = []
        all_gnn_sequences = []
        all_gnn_targets = []
        
        # Process each training match
        for i, (ball_df, home_df, away_df, passes, home_pass_matrix, away_pass_matrix) in enumerate(zip(
                train_ball_dfs, train_home_dfs, train_away_dfs, 
                train_passes, train_home_matrices, train_away_matrices)):
            
            print(f"\nProcessing training match {train_match_nums[i]}...")
            
            # Preprocess data for LSTM
            print("Preprocessing data for LSTM...")
            lstm_X, lstm_y, scaler = preprocess_data_lstm(
                home_df, away_df, ball_df, 
                n_steps=args.sequence_length, n_future=args.prediction_steps
            )
            
            # Prepare pass features
            print("Preparing pass features...")
            pass_features = prepare_pass_features(
                passes, home_df, away_df, ball_df, 
                home_pass_matrix, away_pass_matrix,
                start_idx=args.sequence_length, 
                num_samples=len(lstm_X)
            )
            
            # Prepare GNN dataset
            print("Preparing GNN dataset...")
            gnn_sequences, gnn_targets = prepare_enhanced_gnn_dataset(
                home_df, away_df, ball_df, 
                home_pass_matrix, away_pass_matrix,
                sequence_length=args.sequence_length, 
                prediction_steps=args.prediction_steps,
                proximity_threshold=args.proximity_threshold,
                pass_prob_threshold=args.pass_prob_threshold
            )
            
            # Add to combined datasets
            all_lstm_X.append(lstm_X)
            all_lstm_y.append(lstm_y)
            all_pass_features.append(pass_features)
            all_gnn_sequences.extend(gnn_sequences)  # Extend for sequences
            all_gnn_targets.append(gnn_targets)
        
        # Combine data from all training matches
        lstm_X = np.concatenate(all_lstm_X, axis=0)
        lstm_y = np.concatenate(all_lstm_y, axis=0)
        pass_features = np.concatenate(all_pass_features, axis=0)
        gnn_targets = np.concatenate(all_gnn_targets, axis=0)
        
        # Process validation data
        print("\nProcessing validation data...")
        val_ball_df, val_home_df, val_away_df = val_ball_dfs[0], val_home_dfs[0], val_away_dfs[0]
        val_passes = val_passes[0]
        val_home_matrix, val_away_matrix = val_home_matrices[0], val_away_matrices[0]
        
        # Preprocess validation data
        val_lstm_X, val_lstm_y, _ = preprocess_data_lstm(
            val_home_df, val_away_df, val_ball_df, 
            n_steps=args.sequence_length, n_future=args.prediction_steps,
            scaler=scaler  # Use the same scaler as training data
        )
        
        val_pass_features = prepare_pass_features(
            val_passes, val_home_df, val_away_df, val_ball_df, 
            val_home_matrix, val_away_matrix,
            start_idx=args.sequence_length, 
            num_samples=len(val_lstm_X)
        )
        
        val_gnn_sequences, val_gnn_targets = prepare_enhanced_gnn_dataset(
            val_home_df, val_away_df, val_ball_df, 
            val_home_matrix, val_away_matrix,
            sequence_length=args.sequence_length, 
            prediction_steps=args.prediction_steps,
            proximity_threshold=args.proximity_threshold,
            pass_prob_threshold=args.pass_prob_threshold
        )
        
        # Train LSTM model
        print("\nTraining LSTM model...")
        # Calculate input size from the training data
        input_size = lstm_X.shape[2]  # Features dimension
        
        # Use validation data if available, otherwise use a portion of training data
        if 'val_lstm_X' in locals() and 'val_lstm_y' in locals():
            X_val = val_lstm_X
            y_val = val_lstm_y
        else:
            # Use 20% of training data as validation
            split_idx = int(len(lstm_X) * 0.8)
            X_val = lstm_X[split_idx:]
            y_val = lstm_y[split_idx:]
            lstm_X = lstm_X[:split_idx]
            lstm_y = lstm_y[:split_idx]
        
        lstm_model, lstm_history = train_lstm_model(
            lstm_X, lstm_y, 
            X_val, y_val,
            input_size,
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            model_save_path=lstm_model_path
        )
        
        # Train GNN model
        print("\nTraining GNN model...")
        # Split data for validation if not already available
        if 'val_gnn_sequences' in locals() and 'val_gnn_targets' in locals():
            # Validation data already available
            pass
        else:
            # Use 20% of training data as validation
            split_idx = int(len(all_gnn_sequences) * 0.8)
            val_gnn_sequences = all_gnn_sequences[split_idx:]
            val_gnn_targets = gnn_targets[split_idx:]
            all_gnn_sequences = all_gnn_sequences[:split_idx]
            gnn_targets = gnn_targets[:split_idx]
        
        gnn_model, gnn_history = train_gnn_model(
            all_gnn_sequences, gnn_targets, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            model_save_path=gnn_model_path
        )
        
        # Train enhanced hybrid model
        print("\nTraining enhanced hybrid model...")
        # Prepare validation data for enhanced model if not already available
        if 'val_pass_features' not in locals():
            # Use the same validation split as for LSTM and GNN
            split_idx = int(len(pass_features) * 0.8)
            val_pass_features = pass_features[split_idx:]
            pass_features = pass_features[:split_idx]
        
        enhanced_model, enhanced_history = train_enhanced_hybrid_model(
            lstm_X, all_gnn_sequences, pass_features, lstm_y, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate,
            model_save_path=enhanced_model_path
        )
        
        # Plot training history
        plot_enhanced_training_history(enhanced_history)
        
    elif args.mode == 'test':
        print("Testing models...")
        
        # Get test data
        test_ball_df = test_ball_dfs[0]
        test_home_df = test_home_dfs[0]
        test_away_df = test_away_dfs[0]
        test_passes_data = test_passes[0]
        test_home_matrix = test_home_matrices[0]
        test_away_matrix = test_away_matrices[0]
        
        # Preprocess test data
        print("\nPreprocessing test data...")
        # We need to load a scaler from training
        import joblib
        scaler_path = os.path.join(args.output_dir, f"lstm_scaler_{model_suffix}.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print("Warning: No scaler found. Using default scaling.")
            scaler = None
        
        test_lstm_X, test_lstm_y, _ = preprocess_data_lstm(
            test_home_df, test_away_df, test_ball_df, 
            n_steps=args.sequence_length, n_future=args.prediction_steps,
            scaler=scaler
        )
        
        test_pass_features = prepare_pass_features(
            test_passes_data, test_home_df, test_away_df, test_ball_df, 
            test_home_matrix, test_away_matrix,
            start_idx=args.sequence_length, 
            num_samples=len(test_lstm_X)
        )
        
        test_gnn_sequences, test_gnn_targets = prepare_enhanced_gnn_dataset(
            test_home_df, test_away_df, test_ball_df, 
            test_home_matrix, test_away_matrix,
            sequence_length=args.sequence_length, 
            prediction_steps=args.prediction_steps,
            proximity_threshold=args.proximity_threshold,
            pass_prob_threshold=args.pass_prob_threshold
        )
        
        print("Loading models...")
        try:
            # Load LSTM model
            print("\nEvaluating LSTM model...")
            # Make sure we're loading with the right map_location
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            lstm_model = torch.load(lstm_model_path, map_location=device)
            if isinstance(lstm_model, dict):
                # If we loaded a state dict instead of a model
                from src.hybrid_lstm_gnn.lstm_model import BallPredictorLSTM
                model = BallPredictorLSTM(input_size=test_lstm_X.shape[2])
                model.load_state_dict(lstm_model)
                lstm_model = model
            lstm_model.eval()  # Set to evaluation mode
            lstm_loss, lstm_predictions = evaluate_lstm_model(
                lstm_model, test_lstm_X, test_lstm_y
            )
            
            # Load GNN model
            print("\nEvaluating GNN model...")
            gnn_model = torch.load(gnn_model_path, map_location=device)
            if isinstance(gnn_model, dict):
                # If we loaded a state dict instead of a model
                from src.hybrid_lstm_gnn.gnn_model import BallPredictorGNN
                # Get the input dimension from the first graph in the first sequence
                input_dim = test_gnn_sequences[0][0].x.shape[1] if len(test_gnn_sequences) > 0 and len(test_gnn_sequences[0]) > 0 else 32
                model = BallPredictorGNN(input_dim=input_dim)
                model.load_state_dict(gnn_model)
                gnn_model = model
            gnn_model.eval()  # Set to evaluation mode
            
            # Safely evaluate GNN model
            try:
                gnn_loss, gnn_predictions = evaluate_gnn_model(
                    gnn_model, test_gnn_sequences, test_gnn_targets
                )
            except Exception as e:
                print(f"Error evaluating GNN model: {e}")
                # Create dummy predictions for GNN model
                gnn_predictions = np.zeros_like(test_lstm_y)
                gnn_loss = float('inf')
            
            # Create a simple pass model for comparison
            print("\nEvaluating pass model...")
            pass_input_size = test_pass_features.shape[1]
            pass_model = torch.nn.Sequential(
                torch.nn.Linear(pass_input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 2)
            )
            pass_tensor = torch.tensor(test_pass_features, dtype=torch.float32)
            pass_model.eval()
            with torch.no_grad():
                pass_predictions = pass_model(pass_tensor).numpy()
            
            # Load the enhanced model
            print("\nEvaluating enhanced hybrid model...")
            try:
                enhanced_model = torch.load(enhanced_model_path, map_location=device)
                if isinstance(enhanced_model, dict):
                    # If we loaded a state dict instead of a model
                    from src.hybrid_lstm_gnn.enhanced_hybrid_model import EnhancedHybridModel
                    model = EnhancedHybridModel(
                        lstm_model=lstm_model,
                        gnn_model=gnn_model,
                        pass_model=pass_model,
                        fusion_input_size=None  # Will be determined dynamically
                    )
                    model.load_state_dict(enhanced_model)
                    enhanced_model = model
                enhanced_model.eval()
                enhanced_loss, enhanced_predictions = evaluate_enhanced_hybrid_model(
                    enhanced_model, test_lstm_X, test_gnn_sequences, test_pass_features, test_lstm_y
                )
            except Exception as e:
                print(f"Error evaluating enhanced model: {e}")
                # Use LSTM predictions as a fallback
                enhanced_predictions = lstm_predictions
                enhanced_loss = lstm_loss
            
            # Compare model performance
            metrics = compare_enhanced_model_performance(
                test_lstm_y, lstm_predictions, gnn_predictions, 
                pass_predictions, enhanced_predictions
            )
            
            # Save predictions for match 4 if in predict mode
            if args.mode == 'predict':
                print("\nSaving predictions for match 4...")
                np.save(os.path.join(args.output_dir, f"match_{args.test_match}_predictions.npy"), enhanced_predictions)
                print(f"Predictions saved to {os.path.join(args.output_dir, f'match_{args.test_match}_predictions.npy')}")
            
            # Visualize predictions
            if args.visualize:
                print("\nVisualizing predictions...")
                # Select a segment for visualization
                start_idx = 100
                end_idx = min(start_idx + 50, len(test_lstm_y))
                
                visualize_enhanced_predictions(
                    test_lstm_y, lstm_predictions, gnn_predictions, 
                    pass_predictions, enhanced_predictions,
                    test_home_df, test_away_df, start_idx, end_idx
                )
        except Exception as e:
            print(f"Error during testing: {e}")
    
    elif args.mode == 'demo':
        print("Running demonstration...")
        
        # Use a subset of data
        start_idx = 1000
        sample_size = 200
        ball_df_subset = ball_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        home_df_subset = home_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        away_df_subset = away_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
        
        # Detect passes in the subset
        passes_subset = detect_passes(
            ball_df_subset, home_df_subset, away_df_subset, 
            possession_radius=2.0, ball_speed_threshold=5.0
        )
        
        # Calculate pass probabilities
        home_pass_matrix = create_pass_probability_matrix(passes_subset, 'home_')
        away_pass_matrix = create_pass_probability_matrix(passes_subset, 'away_')
        
        # Create enhanced graph for visualization
        test_idx = 50
        enhanced_graph = create_enhanced_graph(
            home_df_subset, away_df_subset, ball_df_subset, test_idx,
            home_pass_matrix, away_pass_matrix,
            proximity_threshold=args.proximity_threshold,
            pass_prob_threshold=args.pass_prob_threshold
        )
        
        # Extract pass features for the test frame
        pass_features = extract_pass_features_for_frame(
            passes_subset, home_df_subset, away_df_subset, ball_df_subset, test_idx,
            home_pass_matrix, away_pass_matrix
        )
        
        # Visualize the enhanced graph
        from src.hybrid_lstm_gnn.enhanced_visualization import visualize_enhanced_graph
        visualize_enhanced_graph(enhanced_graph, title="Enhanced Graph with Pass Probabilities")
        
        print(f"Pass features for frame {test_idx}:")
        print(pass_features)
        
        print("\nDemonstration complete! Check the output directory for visualizations.")
    
    print("\nEnhanced hybrid model process completed!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
