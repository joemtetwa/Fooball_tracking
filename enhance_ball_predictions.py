import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
import time
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import load_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance Ball Coordinate Predictions with Player Influence')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_ball_predictions.csv',
                        help='Path to the predictions CSV file')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to enhance predictions for')
    parser.add_argument('--output_file', type=str, default='predictions/match_4_enhanced_predictions.csv',
                        help='Path to save enhanced predictions')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize enhanced predictions')
    parser.add_argument('--influence_radius', type=float, default=10.0,
                        help='Radius of player influence on the ball (meters)')
    parser.add_argument('--possession_threshold', type=float, default=2.0,
                        help='Distance threshold for ball possession (meters)')
    parser.add_argument('--max_ball_speed', type=float, default=30.0,
                        help='Maximum ball speed (meters/frame)')
    parser.add_argument('--ball_inertia', type=float, default=0.7,
                        help='Ball inertia factor (0-1), higher means more momentum')
    
    return parser.parse_args()

def calculate_player_influence(player_positions, ball_pos, influence_radius=10.0, team_weights=None):
    """
    Calculate the influence of each player on the ball based on proximity.
    
    Args:
        player_positions: List of player positions [(x1, y1), (x2, y2), ...]
        ball_pos: Ball position (x, y)
        influence_radius: Radius of player influence
        team_weights: Optional weights for home and away teams [home_weight, away_weight]
        
    Returns:
        influence_vector: Vector representing the combined influence of all players
    """
    if len(player_positions) == 0:
        return np.zeros(2)
    
    # Default team weights (equal influence)
    if team_weights is None:
        team_weights = [1.0, 1.0]
    
    # Convert to numpy arrays
    player_positions = np.array(player_positions)
    ball_pos = np.array(ball_pos)
    
    # Calculate distances from ball to each player
    distances = np.array([np.linalg.norm(pos - ball_pos) for pos in player_positions])
    
    # Calculate influence based on inverse square of distance (with cutoff at influence_radius)
    influences = np.zeros_like(distances)
    for i, dist in enumerate(distances):
        if dist < influence_radius:
            # Inverse square law with smooth falloff
            influences[i] = 1.0 / (1.0 + dist**2)
        else:
            influences[i] = 0.0
    
    # Apply team weights (first half of players are home, second half are away)
    num_players = len(player_positions)
    num_home = num_players // 2
    
    for i in range(num_players):
        if i < num_home:
            influences[i] *= team_weights[0]  # Home team weight
        else:
            influences[i] *= team_weights[1]  # Away team weight
    
    # Calculate direction vectors from ball to each player
    directions = np.zeros((num_players, 2))
    for i, pos in enumerate(player_positions):
        direction = pos - ball_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            directions[i] = direction / norm
    
    # Calculate the weighted sum of direction vectors
    influence_vector = np.zeros(2)
    for i in range(num_players):
        influence_vector += directions[i] * influences[i]
    
    return influence_vector

def determine_possession(player_positions, ball_pos, possession_threshold=2.0):
    """
    Determine which player has possession of the ball.
    
    Args:
        player_positions: List of player positions [(x1, y1), (x2, y2), ...]
        ball_pos: Ball position (x, y)
        possession_threshold: Distance threshold for ball possession
        
    Returns:
        player_idx: Index of player with possession, or None if no player has possession
        is_home: True if home team has possession, False if away team
    """
    if len(player_positions) == 0:
        return None, False
    
    # Calculate distances from ball to each player
    distances = np.array([np.linalg.norm(np.array(pos) - np.array(ball_pos)) for pos in player_positions])
    
    # Find the closest player
    closest_idx = np.argmin(distances)
    min_distance = distances[closest_idx]
    
    # Check if the closest player is within the possession threshold
    if min_distance <= possession_threshold:
        # Determine if the player is from the home team (first half of players) or away team
        num_players = len(player_positions)
        is_home = closest_idx < num_players // 2
        return closest_idx, is_home
    else:
        return None, False

def enhance_ball_predictions(predictions, home_df, away_df, args):
    """
    Enhance ball coordinate predictions by incorporating player influence.
    
    Args:
        predictions: DataFrame with predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        args: Command line arguments
        
    Returns:
        enhanced_predictions: DataFrame with enhanced ball coordinates
    """
    print("Enhancing ball predictions with player influence...")
    
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
    
    # Create copy of predictions for enhancement
    enhanced_predictions = predictions.copy()
    
    # Initialize ball velocity
    ball_velocity = np.zeros(2)
    
    # Process each frame
    for i in range(len(predictions) - 1):
        # Get current ball position
        ball_pos = np.array([predictions.iloc[i]['ball_x'], predictions.iloc[i]['ball_y']])
        
        # Scale to meters (assuming the field is 105x68 meters)
        ball_pos_m = ball_pos / 10.0
        
        # Get player positions for this frame
        player_positions = []
        
        # Add home player positions
        for x_col, y_col in zip(home_x_cols, home_y_cols):
            if i < len(home_df):
                x = home_df.iloc[i][x_col]
                y = home_df.iloc[i][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    player_positions.append((x, y))
        
        # Add away player positions
        for x_col, y_col in zip(away_x_cols, away_y_cols):
            if i < len(away_df):
                x = away_df.iloc[i][x_col]
                y = away_df.iloc[i][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    player_positions.append((x, y))
        
        # Determine possession
        player_with_possession, is_home_possession = determine_possession(
            player_positions, ball_pos_m, args.possession_threshold
        )
        
        # Calculate player influence
        # If a player has possession, increase their team's influence
        team_weights = [1.0, 1.0]  # Default equal weights
        if player_with_possession is not None:
            if is_home_possession:
                team_weights = [2.0, 0.5]  # Home team has more influence
            else:
                team_weights = [0.5, 2.0]  # Away team has more influence
        
        influence_vector = calculate_player_influence(
            player_positions, ball_pos_m, args.influence_radius, team_weights
        )
        
        # Apply ball inertia (momentum)
        ball_velocity = args.ball_inertia * ball_velocity + (1 - args.ball_inertia) * influence_vector
        
        # Limit ball speed
        speed = np.linalg.norm(ball_velocity)
        if speed > args.max_ball_speed:
            ball_velocity = ball_velocity * (args.max_ball_speed / speed)
        
        # Update ball position
        new_ball_pos_m = ball_pos_m + ball_velocity
        
        # Scale back to original units
        new_ball_pos = new_ball_pos_m * 10.0
        
        # Update enhanced predictions
        enhanced_predictions.iloc[i+1, enhanced_predictions.columns.get_loc('ball_x')] = new_ball_pos[0]
        enhanced_predictions.iloc[i+1, enhanced_predictions.columns.get_loc('ball_y')] = new_ball_pos[1]
    
    return enhanced_predictions

def visualize_enhanced_predictions(original_predictions, enhanced_predictions, home_df, away_df, start_idx=0, num_frames=20):
    """
    Visualize original and enhanced ball coordinate predictions.
    
    Args:
        original_predictions: DataFrame with original predicted ball coordinates
        enhanced_predictions: DataFrame with enhanced predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        start_idx: Starting frame index
        num_frames: Number of frames to visualize
    """
    print("Visualizing enhanced predictions...")
    
    # Create output directory for visualizations
    os.makedirs('enhanced_visualizations', exist_ok=True)
    
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
    for i in range(min(num_frames, len(enhanced_predictions) - start_idx)):
        frame_idx = start_idx + i
        
        # Get original and enhanced ball positions
        orig_ball_x = original_predictions.iloc[frame_idx]['ball_x'] / 10.0  # Scale to meters
        orig_ball_y = original_predictions.iloc[frame_idx]['ball_y'] / 10.0  # Scale to meters
        
        enh_ball_x = enhanced_predictions.iloc[frame_idx]['ball_x'] / 10.0  # Scale to meters
        enh_ball_y = enhanced_predictions.iloc[frame_idx]['ball_y'] / 10.0  # Scale to meters
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
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
            if frame_idx < len(home_df):
                x = home_df.iloc[frame_idx][x_col]
                y = home_df.iloc[frame_idx][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    plt.plot(x, y, 'bo', markersize=8)
                    plt.text(x + 1, y + 1, x_col.replace('home_', '').replace('_x', ''), fontsize=8)
                    
                    # Draw influence radius for visualization
                    influence_circle = plt.Circle((x, y), 10.0, fill=False, color='b', alpha=0.2)
                    plt.gca().add_patch(influence_circle)
        
        # Draw away players
        for x_col, y_col in zip(away_x_cols, away_y_cols):
            if frame_idx < len(away_df):
                x = away_df.iloc[frame_idx][x_col]
                y = away_df.iloc[frame_idx][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    plt.plot(x, y, 'ro', markersize=8)
                    plt.text(x + 1, y + 1, x_col.replace('away_', '').replace('_x', ''), fontsize=8)
                    
                    # Draw influence radius for visualization
                    influence_circle = plt.Circle((x, y), 10.0, fill=False, color='r', alpha=0.2)
                    plt.gca().add_patch(influence_circle)
        
        # Draw original ball position
        plt.plot(orig_ball_x, orig_ball_y, 'yo', markersize=10, label='Original Prediction')
        
        # Draw enhanced ball position
        plt.plot(enh_ball_x, enh_ball_y, 'go', markersize=10, label='Enhanced Prediction')
        
        # Draw line connecting original and enhanced positions
        plt.plot([orig_ball_x, enh_ball_x], [orig_ball_y, enh_ball_y], 'k--', alpha=0.5)
        
        # Add title and legend
        plt.title(f'Frame {frame_idx}: Ball Position Comparison')
        plt.legend(loc='upper left')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(f'enhanced_visualizations/frame_{frame_idx:04d}.png')
        plt.close()
    
    print(f"Saved {min(num_frames, len(enhanced_predictions) - start_idx)} visualizations to 'enhanced_visualizations' directory")

def create_ball_trajectory_animation(original_predictions, enhanced_predictions, home_df, away_df, start_idx=0, num_frames=100):
    """
    Create an animation showing the ball trajectory for both original and enhanced predictions.
    
    Args:
        original_predictions: DataFrame with original predicted ball coordinates
        enhanced_predictions: DataFrame with enhanced predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        start_idx: Starting frame index
        num_frames: Number of frames to visualize
    """
    print("Creating ball trajectory animation...")
    
    # Create output directory
    os.makedirs('animations', exist_ok=True)
    
    # Set field dimensions
    field_length = 105  # meters
    field_width = 68    # meters
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
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
    
    # Get trajectory points
    end_idx = min(start_idx + num_frames, len(original_predictions))
    
    # Scale to meters
    orig_x = original_predictions.iloc[start_idx:end_idx]['ball_x'].values / 10.0
    orig_y = original_predictions.iloc[start_idx:end_idx]['ball_y'].values / 10.0
    
    enh_x = enhanced_predictions.iloc[start_idx:end_idx]['ball_x'].values / 10.0
    enh_y = enhanced_predictions.iloc[start_idx:end_idx]['ball_y'].values / 10.0
    
    # Plot trajectories
    plt.plot(orig_x, orig_y, 'y-', alpha=0.7, linewidth=2, label='Original Prediction')
    plt.plot(enh_x, enh_y, 'g-', alpha=0.7, linewidth=2, label='Enhanced Prediction')
    
    # Add markers at regular intervals
    marker_interval = max(1, num_frames // 10)
    plt.plot(orig_x[::marker_interval], orig_y[::marker_interval], 'yo', markersize=8)
    plt.plot(enh_x[::marker_interval], enh_y[::marker_interval], 'go', markersize=8)
    
    # Add title and legend
    plt.title(f'Ball Trajectory Comparison (Frames {start_idx}-{end_idx-1})')
    plt.legend(loc='upper left')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(f'animations/ball_trajectory_{start_idx}_{end_idx-1}.png')
    plt.close()
    
    print(f"Saved ball trajectory animation to 'animations/ball_trajectory_{start_idx}_{end_idx-1}.png'")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load match data
    print(f"Loading match {args.match_num} data...")
    _, home_df, away_df = load_data(args.match_num, prediction_mode=True)
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_file}...")
    predictions = pd.read_csv(args.predictions_file)
    
    # Enhance predictions
    enhanced_predictions = enhance_ball_predictions(predictions, home_df, away_df, args)
    
    # Save enhanced predictions
    print(f"Saving enhanced predictions to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    enhanced_predictions.to_csv(args.output_file, index=False)
    
    # Create ball trajectory animation
    create_ball_trajectory_animation(predictions, enhanced_predictions, home_df, away_df, start_idx=1000, num_frames=100)
    
    # Visualize enhanced predictions if requested
    if args.visualize:
        visualize_enhanced_predictions(predictions, enhanced_predictions, home_df, away_df, start_idx=1000, num_frames=20)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
