import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def load_data(match_num):
    """Load ball and player data for a specific match."""
    # Define file paths
    home_file = f"Data/match_{match_num}/Home.xlsx"
    away_file = f"Data/match_{match_num}/Away.xlsx"
    
    print(f"Reading {home_file}")
    home_df = pd.read_excel(home_file)
    
    print(f"Reading {away_file}")
    away_df = pd.read_excel(away_file)
    
    # Extract ball data from home file (ball data is identical in both files)
    ball_cols = ['Time', 'ball_x', 'ball_y']
    ball_df = home_df[ball_cols].copy()
    
    # Print data shapes
    print(f"Ball coordinates shape: {ball_df.shape}")
    print(f"Home players shape: {home_df.shape}")
    print(f"Away players shape: {away_df.shape}")
    
    # Handle NaN values
    ball_df = ball_df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    home_df = home_df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    away_df = away_df.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    
    return ball_df, home_df, away_df

def extract_player_coordinates(home_df, away_df):
    """Extract player coordinates from dataframes, removing non-coordinate columns."""
    # Filter home player coordinates
    home_coords_cols = [col for col in home_df.columns if ('_x' in col or '_y' in col) and 'home_' in col]
    home_coords = home_df[home_coords_cols]
    
    # Filter away player coordinates
    away_coords_cols = [col for col in away_df.columns if ('_x' in col or '_y' in col) and 'away_' in col]
    away_coords = away_df[away_coords_cols]
    
    return home_coords, away_coords

def get_player_positions(home_df, away_df, idx):
    """Extract all player positions at a given index."""
    # Get home player positions
    home_pos = []
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            x = home_df.iloc[idx][x_col]
            y = home_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                home_pos.append((player_id, x, y))
    
    # Get away player positions
    away_pos = []
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            x = away_df.iloc[idx][x_col]
            y = away_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                away_pos.append((player_id, x, y))
    
    return home_pos, away_pos

def apply_player_proximity(predicted_pos, home_players, away_players, proximity_threshold=15.0):
    """Adjust predicted ball position based on player proximity."""
    all_players = home_players + away_players
    
    # Calculate distances to all players
    distances = []
    for player in all_players:
        player_id, player_x, player_y = player
        dist = np.sqrt((predicted_pos[0] - player_x)**2 + (predicted_pos[1] - player_y)**2)
        distances.append((player, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    # If the ball is very close to a player, snap to player position
    if distances and distances[0][1] < proximity_threshold:
        closest_player, dist = distances[0]
        player_id, player_x, player_y = closest_player
        
        # Weighted average based on distance
        weight = 1.0 - (dist / proximity_threshold)
        adjusted_x = (weight * player_x) + ((1.0 - weight) * predicted_pos[0])
        adjusted_y = (weight * player_y) + ((1.0 - weight) * predicted_pos[1])
        
        print(f"Adjusting prediction: Original ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}), "
              f"Closest player {player_id} at dist {dist:.2f}m, "
              f"Adjusted ({adjusted_x:.2f}, {adjusted_y:.2f})")
        
        return np.array([adjusted_x, adjusted_y])
    
    return predicted_pos

def preprocess_data_lstm(home_df, away_df, ball_df, n_steps=5, n_future=1):
    """Preprocess data for LSTM model."""
    # Extract player coordinates
    home_coords, away_coords = extract_player_coordinates(home_df, away_df)
    
    # Combine all features
    features = pd.concat([ball_df[['ball_x', 'ball_y']], home_coords, away_coords], axis=1)
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - n_steps - n_future + 1):
        X.append(scaled_features[i:i+n_steps])
        y.append(scaled_features[i+n_steps:i+n_steps+n_future, :2])  # Only predict ball coordinates
    
    return np.array(X), np.array(y), scaler

def create_demo_visualization(proximity_threshold=15.0):
    """Create a visualization demonstrating player proximity influence."""
    # Create a simple field
    field_length = 105
    field_width = 68
    
    # Create a grid of points
    grid_size = 50
    x = np.linspace(0, field_length, grid_size)
    y = np.linspace(0, field_width, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Create some example player positions
    home_players = [
        ('home_1', 30, 34),
        ('home_2', 60, 40),
        ('home_3', 75, 20),
    ]
    
    away_players = [
        ('away_1', 45, 30),
        ('away_2', 20, 50),
        ('away_3', 80, 45),
    ]
    
    # Calculate influence at each grid point
    influence = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            point = np.array([X[i, j], Y[i, j]])
            adjusted = apply_player_proximity(point, home_players, away_players, proximity_threshold)
            # Calculate the magnitude of adjustment
            influence[i, j] = np.sqrt(np.sum((point - adjusted)**2))
    
    # Plot the field
    plt.figure(figsize=(12, 8))
    plt.contourf(X, Y, influence, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Player Influence (m)')
    
    # Plot field boundaries
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Plot player positions
    for player in home_players:
        player_id, x, y = player
        plt.plot(x, y, 'bo', markersize=10)
        plt.text(x + 1, y + 1, player_id.split('_')[1], fontsize=10)
        
        # Draw influence circle
        influence_circle = plt.Circle((x, y), proximity_threshold, fill=False, color='b', alpha=0.3)
        plt.gca().add_patch(influence_circle)
    
    for player in away_players:
        player_id, x, y = player
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x + 1, y + 1, player_id.split('_')[1], fontsize=10)
        
        # Draw influence circle
        influence_circle = plt.Circle((x, y), proximity_threshold, fill=False, color='r', alpha=0.3)
        plt.gca().add_patch(influence_circle)
    
    plt.title('Player Proximity Influence on Ball Position')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.savefig('player_influence_visualization.png')
    plt.close()
    
    print("Player influence visualization saved to player_influence_visualization.png")

def visualize_frame_with_predictions(ball_df, home_df, away_df, idx, predictions=None, title="Frame Visualization"):
    """Visualize a single frame with ball and player positions, and optional predictions."""
    plt.figure(figsize=(12, 8))
    
    # Get ball position
    ball_x = ball_df.iloc[idx]['ball_x']
    ball_y = ball_df.iloc[idx]['ball_y']
    
    # Get player positions
    home_pos, away_pos = get_player_positions(home_df, away_df, idx)
    
    # Plot field (assuming standard dimensions)
    field_length = 105
    field_width = 68
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Plot home players
    for player in home_pos:
        player_id, x, y = player
        plt.plot(x, y, 'bo', markersize=10)
        plt.text(x + 1, y + 1, player_id.split('_')[1], fontsize=8)
    
    # Plot away players
    for player in away_pos:
        player_id, x, y = player
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x + 1, y + 1, player_id.split('_')[1], fontsize=8)
    
    # Plot ball
    plt.plot(ball_x, ball_y, 'ko', markersize=8)
    plt.text(ball_x + 1, ball_y + 1, 'Ball', fontsize=10)
    
    # Plot predictions if provided
    if predictions is not None:
        for i, pred in enumerate(predictions):
            plt.plot(pred[0], pred[1], 'go', markersize=8)
            plt.text(pred[0] + 1, pred[1] + 1, f'Pred {i+1}', fontsize=8)
            
            # Draw line from actual to prediction
            plt.plot([ball_x, pred[0]], [ball_y, pred[1]], 'g--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f'frame_{idx}_visualization.png')
    plt.close()
    
    print(f"Visualization of frame {idx} saved to frame_{idx}_visualization.png")

def demonstrate_player_proximity():
    """Demonstrate the effect of player proximity on ball predictions."""
    # Load data
    print("Loading data...")
    ball_df, home_df, away_df = load_data(0)
    
    # Use a subset of data
    start_idx = 1000
    sample_size = 100
    ball_df = ball_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
    home_df = home_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
    away_df = away_df.iloc[start_idx:start_idx+sample_size].reset_index(drop=True)
    
    # Create player influence visualization
    create_demo_visualization(proximity_threshold=15.0)
    
    # Choose a frame to visualize
    test_idx = 50
    
    # Get player positions
    home_pos, away_pos = get_player_positions(home_df, away_df, test_idx)
    
    # Get actual ball position
    actual_ball = np.array([ball_df.iloc[test_idx]['ball_x'], ball_df.iloc[test_idx]['ball_y']])
    
    # Create some simulated predictions (offset from actual position)
    offsets = [
        np.array([5.0, 3.0]),
        np.array([-4.0, 2.0]),
        np.array([2.0, -6.0])
    ]
    
    raw_predictions = [actual_ball + offset for offset in offsets]
    
    # Apply player proximity to adjust predictions
    adjusted_predictions = [
        apply_player_proximity(pred, home_pos, away_pos, proximity_threshold=15.0)
        for pred in raw_predictions
    ]
    
    # Visualize the frame with raw predictions
    visualize_frame_with_predictions(
        ball_df, home_df, away_df, test_idx, 
        predictions=raw_predictions,
        title=f"Frame {test_idx} - Raw Predictions"
    )
    
    # Visualize the frame with adjusted predictions
    visualize_frame_with_predictions(
        ball_df, home_df, away_df, test_idx, 
        predictions=adjusted_predictions,
        title=f"Frame {test_idx} - Adjusted Predictions with Player Proximity"
    )

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run demonstration
    demonstrate_player_proximity()
