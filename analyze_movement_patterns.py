import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import networkx as nx
from collections import defaultdict, Counter
import argparse
import time
from matplotlib.patches import Circle, Rectangle

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import load_data
from pass_analysis import (
    detect_passes, calculate_pass_probabilities, visualize_pass_network,
    calculate_average_player_positions, analyze_ball_possession_sequences
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Movement Patterns from Enhanced Ball Predictions')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_enhanced_predictions.csv',
                        help='Path to the enhanced predictions CSV file')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis/movement_patterns',
                        help='Directory to save analysis results')
    parser.add_argument('--start_frame', type=int, default=1000,
                        help='Starting frame for analysis')
    parser.add_argument('--num_frames', type=int, default=10000,
                        help='Number of frames to analyze')
    parser.add_argument('--possession_radius', type=float, default=5.0,
                        help='Distance threshold for ball possession (meters)')
    parser.add_argument('--ball_speed_threshold', type=float, default=7.0,
                        help='Speed threshold for pass detection (meters/second)')
    parser.add_argument('--shot_speed_threshold', type=float, default=15.0,
                        help='Speed threshold for shot detection (meters/second)')
    parser.add_argument('--dribble_max_speed', type=float, default=6.0,
                        help='Maximum speed for dribble detection (meters/second)')
    parser.add_argument('--goal_width', type=float, default=7.32,
                        help='Width of goal in meters')
    
    return parser.parse_args()

def calculate_ball_velocity(ball_positions, frames_per_second=10):
    """
    Calculate ball velocity from position data.
    
    Parameters:
    -----------
    ball_positions : DataFrame
        DataFrame containing ball x, y coordinates
    frames_per_second : int
        Frame rate of the tracking data
    
    Returns:
    --------
    DataFrame with ball velocity columns added
    """
    # Create a copy of the input data
    ball_df = ball_positions.copy()
    
    # Calculate velocity (displacement between frames)
    ball_df['ball_vx'] = ball_df['ball_x'].diff() * frames_per_second
    ball_df['ball_vy'] = ball_df['ball_y'].diff() * frames_per_second
    
    # Calculate speed
    ball_df['ball_speed'] = np.sqrt(ball_df['ball_vx']**2 + ball_df['ball_vy']**2)
    
    # Calculate acceleration
    ball_df['ball_ax'] = ball_df['ball_vx'].diff() * frames_per_second
    ball_df['ball_ay'] = ball_df['ball_vy'].diff() * frames_per_second
    ball_df['ball_accel'] = np.sqrt(ball_df['ball_ax']**2 + ball_df['ball_ay']**2)
    
    # Calculate direction (angle in radians)
    ball_df['ball_direction'] = np.arctan2(ball_df['ball_vy'], ball_df['ball_vx'])
    
    # Fill NaN values (first row of diff)
    ball_df.fillna(0, inplace=True)
    
    return ball_df

def create_enhanced_ball_df(predictions, start_frame, num_frames):
    """
    Create an enhanced ball DataFrame from predictions.
    
    Parameters:
    -----------
    predictions : DataFrame
        DataFrame containing predicted ball coordinates
    start_frame, num_frames : int
        Range of frames to analyze
    
    Returns:
    --------
    DataFrame with ball position and velocity
    """
    # Extract relevant frames
    end_frame = min(start_frame + num_frames, len(predictions))
    selected_frames = range(start_frame, end_frame)
    
    # Create ball DataFrame
    ball_df = pd.DataFrame({
        'frame': list(selected_frames),
        'ball_x': [predictions.iloc[i]['ball_x'] / 100.0 for i in selected_frames],  # Scale to meters
        'ball_y': [predictions.iloc[i]['ball_y'] / 100.0 for i in selected_frames]   # Scale to meters
    })
    
    # Add velocity information
    ball_df = calculate_ball_velocity(ball_df)
    
    return ball_df

def detect_shots(ball_df, home_df, away_df, field_length=105, field_width=68, 
                goal_width=7.32, shot_speed_threshold=15.0, possession_radius=5.0):
    """
    Detect shots based on ball trajectory and speed.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame with ball position and velocity
    home_df, away_df : DataFrame
        DataFrames with player positions
    field_length, field_width : float
        Field dimensions in meters
    goal_width : float
        Width of goal in meters
    shot_speed_threshold : float
        Minimum ball speed to be considered a shot
    possession_radius : float
        Distance threshold for determining ball possession
    
    Returns:
    --------
    shots : list
        List of tuples (player_id, frame, shot_type, on_target)
    """
    shots = []
    goal_posts = {
        'home': {'left': (0, (field_width - goal_width) / 2), 'right': (0, (field_width + goal_width) / 2)},
        'away': {'left': (field_length, (field_width - goal_width) / 2), 'right': (field_length, (field_width + goal_width) / 2)}
    }
    
    for i in range(1, len(ball_df) - 10):  # Skip first frame and ensure we have at least 10 frames afterward
        # Check for high ball speed
        if ball_df.iloc[i]['ball_speed'] >= shot_speed_threshold:
            # Get current and previous frames
            curr_frame = ball_df.iloc[i]
            prev_frame = ball_df.iloc[i-1]
            
            # Check for player possession in previous frame
            player_positions = []
            frame_idx = curr_frame['frame']
            
            # Get home player positions
            for col in home_df.columns:
                if '_x' in col and 'home_' in col:
                    player_id = col.replace('_x', '')
                    y_col = f"{player_id}_y"
                    if y_col in home_df.columns and frame_idx < len(home_df):
                        x = home_df.iloc[frame_idx][col]
                        y = home_df.iloc[frame_idx][y_col]
                        if not np.isnan(x) and not np.isnan(y):
                            player_positions.append((player_id, x, y, True))  # True = home team
            
            # Get away player positions
            for col in away_df.columns:
                if '_x' in col and 'away_' in col:
                    player_id = col.replace('_x', '')
                    y_col = f"{player_id}_y"
                    if y_col in away_df.columns and frame_idx < len(away_df):
                        x = away_df.iloc[frame_idx][col]
                        y = away_df.iloc[frame_idx][y_col]
                        if not np.isnan(x) and not np.isnan(y):
                            player_positions.append((player_id, x, y, False))  # False = away team
            
            # Find closest player to the ball
            closest_player = None
            min_dist = float('inf')
            for player_id, x, y, is_home in player_positions:
                dist = np.sqrt((x - prev_frame['ball_x'])**2 + (y - prev_frame['ball_y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player = (player_id, is_home)
            
            # Check if any player was close enough to possess the ball
            if closest_player and min_dist <= possession_radius:
                player_id, is_home = closest_player
                
                # Determine target goal (opposite team's goal)
                target_goal = 'away' if is_home else 'home'
                
                # Check ball direction (for home team: positive x direction, for away team: negative x direction)
                correct_direction = False
                if (is_home and curr_frame['ball_vx'] > 0) or (not is_home and curr_frame['ball_vx'] < 0):
                    correct_direction = True
                
                if correct_direction:
                    # Project ball trajectory to see if it would hit the goal
                    on_target = False
                    slope = curr_frame['ball_vy'] / curr_frame['ball_vx'] if curr_frame['ball_vx'] != 0 else float('inf')
                    
                    # For home team shooting at away goal (at x = field_length)
                    if is_home:
                        # Calculate where ball would cross goal line
                        if curr_frame['ball_vx'] > 0:  # Ensure ball is moving toward goal
                            y_intersection = curr_frame['ball_y'] + slope * (field_length - curr_frame['ball_x'])
                            if (field_width - goal_width) / 2 <= y_intersection <= (field_width + goal_width) / 2:
                                on_target = True
                    
                    # For away team shooting at home goal (at x = 0)
                    else:
                        # Calculate where ball would cross goal line
                        if curr_frame['ball_vx'] < 0:  # Ensure ball is moving toward goal
                            y_intersection = curr_frame['ball_y'] + slope * (0 - curr_frame['ball_x'])
                            if (field_width - goal_width) / 2 <= y_intersection <= (field_width + goal_width) / 2:
                                on_target = True
                    
                    # Store shot information
                    shot_type = "On Target" if on_target else "Off Target"
                    shots.append((player_id, frame_idx, shot_type, on_target))
    
    return shots

def detect_dribbles(ball_df, home_df, away_df, possession_radius=5.0, dribble_max_speed=6.0, min_dribble_frames=5):
    """
    Detect dribbling sequences based on continuous ball possession at moderate speed.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame with ball position and velocity
    home_df, away_df : DataFrame
        DataFrames with player positions
    possession_radius : float
        Distance threshold for determining ball possession
    dribble_max_speed : float
        Maximum ball speed to be considered a dribble
    min_dribble_frames : int
        Minimum number of consecutive frames to be considered a dribble
    
    Returns:
    --------
    dribbles : list
        List of tuples (player_id, start_frame, end_frame, distance)
    """
    dribbles = []
    current_dribble = None
    
    for i in range(len(ball_df)):
        frame_idx = ball_df.iloc[i]['frame']
        ball_x = ball_df.iloc[i]['ball_x']
        ball_y = ball_df.iloc[i]['ball_y']
        ball_speed = ball_df.iloc[i]['ball_speed']
        
        # Skip if ball is moving too fast
        if ball_speed > dribble_max_speed:
            if current_dribble:
                player_id, start_frame, positions = current_dribble
                
                # Check if dribble is long enough
                if len(positions) >= min_dribble_frames:
                    # Calculate total dribbling distance
                    total_distance = 0
                    for j in range(1, len(positions)):
                        x1, y1 = positions[j-1]
                        x2, y2 = positions[j]
                        segment_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        total_distance += segment_dist
                    
                    dribbles.append((player_id, start_frame, frame_idx-1, total_distance))
                
                current_dribble = None
            continue
        
        # Get player positions
        player_positions = []
        
        # Get home player positions
        for col in home_df.columns:
            if '_x' in col and 'home_' in col:
                player_id = col.replace('_x', '')
                y_col = f"{player_id}_y"
                if y_col in home_df.columns and frame_idx < len(home_df):
                    x = home_df.iloc[frame_idx][col]
                    y = home_df.iloc[frame_idx][y_col]
                    if not np.isnan(x) and not np.isnan(y):
                        player_positions.append((player_id, x, y))
        
        # Get away player positions
        for col in away_df.columns:
            if '_x' in col and 'away_' in col:
                player_id = col.replace('_x', '')
                y_col = f"{player_id}_y"
                if y_col in away_df.columns and frame_idx < len(away_df):
                    x = away_df.iloc[frame_idx][col]
                    y = away_df.iloc[frame_idx][y_col]
                    if not np.isnan(x) and not np.isnan(y):
                        player_positions.append((player_id, x, y))
        
        # Find closest player to the ball
        closest_player = None
        min_dist = float('inf')
        for player_id, x, y in player_positions:
            dist = np.sqrt((x - ball_x)**2 + (y - ball_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_player = player_id
        
        # Check if any player is possessing the ball
        if closest_player and min_dist <= possession_radius:
            # Continue existing dribble or start a new one
            if current_dribble:
                last_player, start_frame, positions = current_dribble
                
                if last_player == closest_player:
                    # Continue the dribble
                    positions.append((ball_x, ball_y))
                else:
                    # Different player has the ball, end current dribble and start a new one
                    if len(positions) >= min_dribble_frames:
                        # Calculate total dribbling distance
                        total_distance = 0
                        for j in range(1, len(positions)):
                            x1, y1 = positions[j-1]
                            x2, y2 = positions[j]
                            segment_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            total_distance += segment_dist
                        
                        dribbles.append((last_player, start_frame, frame_idx-1, total_distance))
                    
                    # Start a new dribble
                    current_dribble = (closest_player, frame_idx, [(ball_x, ball_y)])
            else:
                # Start a new dribble
                current_dribble = (closest_player, frame_idx, [(ball_x, ball_y)])
        else:
            # No player has possession, end any current dribble
            if current_dribble:
                player_id, start_frame, positions = current_dribble
                
                if len(positions) >= min_dribble_frames:
                    # Calculate total dribbling distance
                    total_distance = 0
                    for j in range(1, len(positions)):
                        x1, y1 = positions[j-1]
                        x2, y2 = positions[j]
                        segment_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        total_distance += segment_dist
                    
                    dribbles.append((player_id, start_frame, frame_idx-1, total_distance))
                
                current_dribble = None
    
    # Handle any ongoing dribble at the end
    if current_dribble:
        player_id, start_frame, positions = current_dribble
        
        if len(positions) >= min_dribble_frames:
            # Calculate total dribbling distance
            total_distance = 0
            for j in range(1, len(positions)):
                x1, y1 = positions[j-1]
                x2, y2 = positions[j]
                segment_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += segment_dist
            
            frame_idx = ball_df.iloc[-1]['frame']
            dribbles.append((player_id, start_frame, frame_idx, total_distance))
    
    return dribbles
