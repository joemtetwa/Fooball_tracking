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
from src.hybrid_lstm_gnn.data_utils import load_data, preprocess_data_lstm, get_player_positions
from src.hybrid_lstm_gnn.lstm_model import BallPredictorLSTM, predict_with_lstm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ball Coordinate Prediction')
    
    parser.add_argument('--model_path', type=str, default='models/lstm_model_train_0,1,2_val_3.pt',
                        help='Path to the trained LSTM model')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to predict')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Sequence length for LSTM')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    
    return parser.parse_args()

def prepare_prediction_data(home_df, away_df, sequence_length=5):
    """
    Prepare data for prediction when ball coordinates are not available.
    
    Args:
        home_df, away_df: DataFrames with player positions
        sequence_length: Length of input sequences
        
    Returns:
        LSTM input data
    """
    print("Preparing prediction data...")
    
    # Extract player positions for each frame
    X = []
    
    for i in range(sequence_length, len(home_df)):
        # Get sequence of player positions
        sequence = []
        
        for j in range(i - sequence_length, i):
            # Get player positions for this frame
            # The get_player_positions function returns a combined array of all player positions
            all_player_positions = get_player_positions(home_df, away_df, j)
            
            # Flatten positions into a feature vector
            features = all_player_positions.flatten().tolist()
            
            # If we have fewer players than expected, pad with zeros
            expected_feature_length = 44  # 22 players * 2 coordinates
            if len(features) < expected_feature_length:
                features.extend([0] * (expected_feature_length - len(features)))
            # If we have more features than expected, truncate
            elif len(features) > expected_feature_length:
                features = features[:expected_feature_length]
                
            sequence.append(features)
        
        X.append(sequence)
    
    # Convert to numpy array
    X = np.array(X)
    
    # Normalize data (simple min-max scaling)
    X_flat = X.reshape(-1, X.shape[-1])
    X_min, X_max = X_flat.min(axis=0), X_flat.max(axis=0)
    X_scaled = (X_flat - X_min) / (X_max - X_min + 1e-10)
    X = X_scaled.reshape(X.shape)
    
    return X

def visualize_predictions(predictions, home_df, away_df, start_idx=0, num_frames=100):
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
    
    # Visualize each frame
    for i in range(min(num_frames, len(predictions) - start_idx)):
        frame_idx = start_idx + i
        
        # Get predicted ball position
        ball_x, ball_y = predictions[frame_idx]
        
        # Get player positions (returns a combined array of all player positions)
        all_player_positions = get_player_positions(home_df, away_df, frame_idx)
        
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
        
        # Determine how many players are from each team (assuming first half are home players)
        num_players = len(all_player_positions)
        num_home_players = num_players // 2
        
        # Draw home players (first half of all_player_positions)
        for i, pos in enumerate(all_player_positions[:num_home_players]):
            plt.plot(pos[0], pos[1], 'bo', markersize=8)
            plt.text(pos[0] + 1, pos[1] + 1, f'H{i+1}', fontsize=8)
        
        # Draw away players (second half of all_player_positions)
        for i, pos in enumerate(all_player_positions[num_home_players:]):
            plt.plot(pos[0], pos[1], 'ro', markersize=8)
            plt.text(pos[0] + 1, pos[1] + 1, f'A{i+1}', fontsize=8)
        
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
    """Main function to predict ball coordinates."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load match data
    print(f"Loading data for match {args.match_num}...")
    ball_df, home_df, away_df = load_data(args.match_num, prediction_mode=True)
    
    # Prepare data for prediction
    X = prepare_prediction_data(home_df, away_df, sequence_length=args.sequence_length)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = torch.load(args.model_path, map_location=device)
    
    # Check if the loaded object is a state dictionary
    if isinstance(model_data, dict):
        print("Loaded state dictionary, creating model instance...")
        from src.hybrid_lstm_gnn.lstm_model import BallPredictorLSTM
        # Create model with appropriate input size
        input_size = X.shape[2]  # Get input size from prepared data
        model = BallPredictorLSTM(input_size=input_size)
        model.load_state_dict(model_data)
    else:
        model = model_data
        
    model.eval()
    
    # Make predictions
    print("Generating predictions...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Save predictions
    output_path = os.path.join(args.output_dir, f"match_{args.match_num}_ball_predictions.npy")
    np.save(output_path, predictions)
    print(f"Saved predictions to {output_path}")
    
    # Save as CSV for easier analysis
    df = pd.DataFrame(predictions, columns=['ball_x', 'ball_y'])
    csv_path = os.path.join(args.output_dir, f"match_{args.match_num}_ball_predictions.csv")
    df.to_csv(csv_path, index=True)
    print(f"Saved predictions as CSV to {csv_path}")
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(predictions, home_df, away_df, start_idx=1000, num_frames=20)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
