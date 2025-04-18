import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                home_pos.append((x, y))
    
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
                away_pos.append((x, y))
    
    # Combine all player positions
    all_players = home_pos + away_pos
    return np.array(all_players)

def test_data_loading():
    """Test loading data from match_0."""
    print("Testing data loading from match_0...")
    
    # Load data
    ball_df, home_df, away_df = load_data(0)
    
    # Print basic information
    print(f"Ball data shape: {ball_df.shape}")
    print(f"Home data shape: {home_df.shape}")
    print(f"Away data shape: {away_df.shape}")
    
    # Print column names
    print("\nBall columns:", list(ball_df.columns))
    
    # Extract player coordinates
    home_coords, away_coords = extract_player_coordinates(home_df, away_df)
    
    print(f"\nHome player coordinate columns: {len(home_coords.columns)}")
    print(f"Away player coordinate columns: {len(away_coords.columns)}")
    
    # Get unique player IDs
    home_player_ids = set([col.split('_x')[0] for col in home_coords.columns if '_x' in col])
    away_player_ids = set([col.split('_x')[0] for col in away_coords.columns if '_x' in col])
    
    print(f"\nUnique home players: {len(home_player_ids)}")
    print(f"Home player IDs: {sorted(list(home_player_ids))}")
    
    print(f"\nUnique away players: {len(away_player_ids)}")
    print(f"Away player IDs: {sorted(list(away_player_ids))}")
    
    # Test getting player positions for a specific frame
    test_idx = 1000  # Choose a frame index
    player_positions = get_player_positions(home_df, away_df, test_idx)
    
    print(f"\nPlayer positions at frame {test_idx}: {len(player_positions)} players")
    
    # Visualize a sample frame
    visualize_frame(ball_df, home_df, away_df, test_idx)
    
    return ball_df, home_df, away_df

def visualize_frame(ball_df, home_df, away_df, idx):
    """Visualize a single frame with ball and player positions."""
    plt.figure(figsize=(12, 8))
    
    # Get ball position
    ball_x = ball_df.iloc[idx]['ball_x']
    ball_y = ball_df.iloc[idx]['ball_y']
    
    # Get player positions
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    home_positions = []
    for x_col in home_player_cols:
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            x = home_df.iloc[idx][x_col]
            y = home_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                home_positions.append((x, y, player_id))
    
    away_positions = []
    for x_col in away_player_cols:
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            x = away_df.iloc[idx][x_col]
            y = away_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                away_positions.append((x, y, player_id))
    
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
    for pos in home_positions:
        plt.plot(pos[0], pos[1], 'bo', markersize=10)
        plt.text(pos[0] + 1, pos[1] + 1, pos[2].split('_')[1], fontsize=8)
    
    # Plot away players
    for pos in away_positions:
        plt.plot(pos[0], pos[1], 'ro', markersize=10)
        plt.text(pos[0] + 1, pos[1] + 1, pos[2].split('_')[1], fontsize=8)
    
    # Plot ball
    plt.plot(ball_x, ball_y, 'ko', markersize=8)
    plt.text(ball_x + 1, ball_y + 1, 'Ball', fontsize=10)
    
    plt.title(f'Frame {idx} - Ball and Player Positions')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('test_frame_visualization.png')
    plt.close()
    
    print(f"Visualization of frame {idx} saved to test_frame_visualization.png")

if __name__ == "__main__":
    ball_df, home_df, away_df = test_data_loading()
