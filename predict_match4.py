import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import load_data, preprocess_data_lstm
from src.hybrid_lstm_gnn.lstm_model import BallPredictorLSTM

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ball Coordinate Prediction for Match 4')
    
    parser.add_argument('--model_path', type=str, default='models/lstm_model_train_0,1,2_val_3.pt',
                        help='Path to the trained LSTM model')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--reference_match', type=int, default=0,
                        help='Reference match to use for feature structure')
    
    return parser.parse_args()

def extract_player_features(home_df, away_df, sequence_length=5):
    """
    Extract player position features from match data.
    
    Args:
        home_df, away_df: DataFrames with player positions
        sequence_length: Length of sequences to create
        
    Returns:
        Numpy array of player features
    """
    print("Extracting player features...")
    
    # Get player position columns
    home_x_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    home_y_cols = [col for col in home_df.columns if '_y' in col and 'home_' in col]
    away_x_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    away_y_cols = [col for col in away_df.columns if '_y' in col and 'away_' in col]
    
    # Sort columns to ensure consistent order
    home_x_cols.sort()
    home_y_cols.sort()
    away_x_cols.sort()
    away_y_cols.sort()
    
    # Create sequences
    sequences = []
    
    for i in range(sequence_length, len(home_df)):
        # Get sequence of frames
        sequence = []
        
        for j in range(i - sequence_length, i):
            # Extract player positions for this frame
            frame_features = []
            
            # Add home player positions
            for x_col, y_col in zip(home_x_cols, home_y_cols):
                x = home_df.iloc[j][x_col]
                y = home_df.iloc[j][y_col]
                frame_features.extend([x, y])
            
            # Add away player positions
            for x_col, y_col in zip(away_x_cols, away_y_cols):
                x = away_df.iloc[j][x_col]
                y = away_df.iloc[j][y_col]
                frame_features.extend([x, y])
            
            sequence.append(frame_features)
        
        sequences.append(sequence)
    
    return np.array(sequences)

def adapt_features_to_model(features, reference_features):
    """
    Adapt features to match the expected input structure of the model.
    
    Args:
        features: Features extracted from match 4
        reference_features: Features from a reference match used for training
        
    Returns:
        Adapted features matching the expected input structure
    """
    print("Adapting features to match model input structure...")
    
    # Get expected feature size
    expected_feature_size = reference_features.shape[2]
    actual_feature_size = features.shape[2]
    
    print(f"Expected feature size: {expected_feature_size}, Actual feature size: {actual_feature_size}")
    
    if actual_feature_size == expected_feature_size:
        # Features already match
        return features
    
    # Create adapted features with the expected size
    adapted_features = np.zeros((features.shape[0], features.shape[1], expected_feature_size))
    
    # Copy available features
    min_size = min(actual_feature_size, expected_feature_size)
    adapted_features[:, :, :min_size] = features[:, :, :min_size]
    
    return adapted_features

def visualize_predictions(predictions, home_df, away_df, start_idx=0, num_frames=20):
    """
    Visualize predicted ball coordinates with player positions.
    
    Args:
        predictions: Predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        start_idx: Starting frame index
        num_frames: Number of frames to visualize
    """
    print("Visualizing predictions...")
    
    # Create output directory for visualizations
    os.makedirs('visualizations', exist_ok=True)
    
    # Set field dimensions
    field_length = 105  # meters
    field_width = 68    # meters
    
    # Get player position columns
    home_x_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    home_y_cols = [col for col in home_df.columns if '_y' in col and 'home_' in col]
    away_x_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    away_y_cols = [col for col in away_df.columns if '_y' in col and 'away_' in col]
    
    # Sort columns to ensure consistent order
    home_x_cols.sort()
    home_y_cols.sort()
    away_x_cols.sort()
    away_y_cols.sort()
    
    # Visualize each frame
    for i in range(min(num_frames, len(predictions) - start_idx)):
        frame_idx = start_idx + i
        
        # Get predicted ball position
        ball_x, ball_y = predictions[frame_idx]
        
        # Create figure
        plt.figure(figsize=(10, 7))
        
        # Draw field
        plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
        plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
        
        # Draw center circle
        center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
        plt.gca().add_patch(center_circle)
        
        # Draw penalty areas
        plt.plot([0, 16.5, 16.5, 0], [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
        plt.plot([field_length, field_length - 16.5, field_length - 16.5, field_length], 
                 [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
        
        # Draw goal areas
        plt.plot([0, 5.5, 5.5, 0], [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
        plt.plot([field_length, field_length - 5.5, field_length - 5.5, field_length], 
                 [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
        
        # Draw penalty spots
        plt.plot(11, field_width/2, 'ko', markersize=3)
        plt.plot(field_length - 11, field_width/2, 'ko', markersize=3)
        
        # Draw home players
        for x_col, y_col in zip(home_x_cols, home_y_cols):
            x = home_df.iloc[frame_idx][x_col]
            y = home_df.iloc[frame_idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                plt.plot(x, y, 'bo', markersize=8)
                plt.text(x + 1, y + 1, x_col.replace('home_', '').replace('_x', ''), fontsize=8)
        
        # Draw away players
        for x_col, y_col in zip(away_x_cols, away_y_cols):
            x = away_df.iloc[frame_idx][x_col]
            y = away_df.iloc[frame_idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                plt.plot(x, y, 'ro', markersize=8)
                plt.text(x + 1, y + 1, x_col.replace('away_', '').replace('_x', ''), fontsize=8)
        
        # Draw predicted ball position
        plt.plot(ball_x, ball_y, 'yo', markersize=10, label='Predicted Ball')
        
        # Add title and legend
        plt.title(f'Frame {frame_idx}: Predicted Ball Position')
        plt.legend(loc='upper left')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f'visualizations/frame_{frame_idx:04d}.png')
        plt.close()
    
    print(f"Saved {min(num_frames, len(predictions) - start_idx)} visualizations to 'visualizations' directory")

def main():
    """Main function to predict ball coordinates for match 4."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load match 4 data
    print("Loading match 4 data...")
    _, home_df4, away_df4 = load_data(4, prediction_mode=True)
    
    # Load reference match data to get feature structure
    print(f"Loading reference match {args.reference_match} data...")
    ball_df_ref, home_df_ref, away_df_ref = load_data(args.reference_match)
    
    # Preprocess reference data to get feature structure
    print("Preprocessing reference data...")
    X_ref, y_ref, _ = preprocess_data_lstm(home_df_ref, away_df_ref, ball_df_ref, n_steps=5)
    
    # Extract player features from match 4
    features = extract_player_features(home_df4, away_df4, sequence_length=5)
    
    # Adapt features to match the expected input structure
    adapted_features = adapt_features_to_model(features, X_ref)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = torch.load(args.model_path, map_location=device)
    
    # Check if the loaded object is a state dictionary
    if isinstance(model_data, dict):
        print("Loaded state dictionary, creating model instance...")
        # Create model with appropriate input size
        input_size = X_ref.shape[2]  # Get input size from reference data
        model = BallPredictorLSTM(input_size=input_size)
        model.load_state_dict(model_data)
    else:
        model = model_data
        
    model.eval()
    
    # Make predictions
    print("Generating predictions...")
    X_tensor = torch.tensor(adapted_features, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Save predictions
    output_path = os.path.join(args.output_dir, "match_4_ball_predictions.npy")
    np.save(output_path, predictions)
    print(f"Saved predictions to {output_path}")
    
    # Save as CSV for easier analysis
    df = pd.DataFrame(predictions, columns=['ball_x', 'ball_y'])
    csv_path = os.path.join(args.output_dir, "match_4_ball_predictions.csv")
    df.to_csv(csv_path, index=True)
    print(f"Saved predictions as CSV to {csv_path}")
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(predictions, home_df4, away_df4, start_idx=1000, num_frames=20)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
