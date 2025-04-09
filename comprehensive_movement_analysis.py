import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import matplotlib.patches as patches
import time
import argparse
from collections import defaultdict, Counter

# Import functions from existing modules
from pass_analysis import (
    load_data,
    detect_passes,
    calculate_pass_probabilities,
    visualize_pass_network,
    calculate_average_player_positions
)

from analyze_movement_patterns import (
    calculate_ball_velocity,
    create_enhanced_ball_df
)

# We're adding our own fixed versions of detect_shots and detect_dribbles functions

# Enhanced version of detect_shots function
def detect_shots(ball_df, home_df, away_df, field_length=105, field_width=68, 
               goal_width=7.32, shot_speed_threshold=12.0, possession_radius=5.0):
    """
    Detect shots on goal based on ball trajectory and speed.
    
    Enhanced version to better detect shots using player influence model and trajectory analysis.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame containing ball coordinates
    home_df, away_df : DataFrame
        DataFrames containing player coordinates
    field_length, field_width : float
        Dimensions of the soccer field
    goal_width : float
        Width of the goal
    shot_speed_threshold : float
        Minimum ball speed to be considered a shot
    possession_radius : float
        Maximum distance between player and ball to be in possession
    
    Returns:
    --------
    shots : list
        List of tuples (player_id, frame, shot_type, on_target)
    """
    # Goals are located at (0, field_width/2) and (field_length, field_width/2)
    home_goal = (0, field_width / 2)
    away_goal = (field_length, field_width / 2)
    
    shots = []
    
    # Check if ball_df has velocity calculated, if not, calculate it
    if 'ball_velocity' not in ball_df.columns:
        ball_df = calculate_ball_velocity(ball_df)
    
    # Calculate and check for potential shots
    for i in range(1, len(ball_df) - 3):  # We need a few frames to analyze trajectory
        # Get ball positions and speed
        curr_ball_x, curr_ball_y = ball_df['ball_x'].iloc[i], ball_df['ball_y'].iloc[i]
        next_ball_x, next_ball_y = ball_df['ball_x'].iloc[i+1], ball_df['ball_y'].iloc[i+1]
        prev_ball_x, prev_ball_y = ball_df['ball_x'].iloc[i-1], ball_df['ball_y'].iloc[i-1]
        
        # Get current ball velocity (if available), otherwise calculate it
        if 'ball_velocity' in ball_df.columns:
            ball_speed = ball_df['ball_velocity'].iloc[i]
        else:
            # Calculate ball speed (using distance between consecutive frames)
            dx = next_ball_x - curr_ball_x
            dy = next_ball_y - curr_ball_y
            ball_speed = np.sqrt(dx**2 + dy**2)
        
        # Detect sudden acceleration in ball speed as potential shots
        if ball_speed >= shot_speed_threshold:
            # Calculate ball direction change - shots often involve a significant direction change
            prev_direction = np.arctan2(curr_ball_y - prev_ball_y, curr_ball_x - prev_ball_x) if i > 0 else 0
            curr_direction = np.arctan2(next_ball_y - curr_ball_y, next_ball_x - curr_ball_x)
            direction_change = abs(curr_direction - prev_direction)
            if direction_change > np.pi:
                direction_change = 2 * np.pi - direction_change
            
            # Check if ball is heading towards either goal
            heading_to_home_goal = curr_ball_x > next_ball_x  # Ball moving towards x=0 (home goal)
            heading_to_away_goal = curr_ball_x < next_ball_x  # Ball moving towards x=105 (away goal)
            
            # Find closest player to previous ball position (likely the shooter)
            min_dist = float('inf')
            closest_player = None
            
            # Check home players
            for col in home_df.columns:
                if '_x' in col and 'home_' in col:
                    player_id = col.replace('_x', '')
                    y_col = f"{player_id}_y"
                    if y_col in home_df.columns:
                        x = home_df[col].iloc[i]  # Direct indexing using i
                        y = home_df[y_col].iloc[i]
                        dist = np.sqrt((x - prev_ball_x)**2 + (y - prev_ball_y)**2)
                        if dist < min_dist and dist <= possession_radius:
                            min_dist = dist
                            closest_player = player_id
            
            # Check away players
            for col in away_df.columns:
                if '_x' in col and 'away_' in col:
                    player_id = col.replace('_x', '')
                    y_col = f"{player_id}_y"
                    if y_col in away_df.columns:
                        x = away_df[col].iloc[i]  # Direct indexing using i
                        y = away_df[y_col].iloc[i]
                        dist = np.sqrt((x - prev_ball_x)**2 + (y - prev_ball_y)**2)
                        if dist < min_dist and dist <= possession_radius:
                            min_dist = dist
                            closest_player = player_id
            
            if closest_player:
                # Determine if this is a valid shot attempt
                shooting_home = closest_player.startswith('home_')
                valid_shot_direction = (shooting_home and heading_to_away_goal) or \
                                     (not shooting_home and heading_to_home_goal)
                
                # Factor in field position - shots typically happen closer to goal
                attack_third = (shooting_home and curr_ball_x > field_length*0.6) or \
                             (not shooting_home and curr_ball_x < field_length*0.4)
                
                # Consider shot if direction is valid or we're in attacking third
                if valid_shot_direction or attack_third:
                    # Determine target goal based on player team
                    target_goal = away_goal if shooting_home else home_goal
                    
                    # Check distance to goal - closer shots are more likely to be deliberate shots
                    dist_to_goal = np.sqrt((target_goal[0] - curr_ball_x)**2 + 
                                         (target_goal[1] - curr_ball_y)**2)
                    
                    # Determine shot type based on position and speed
                    if ball_speed > shot_speed_threshold * 1.5:
                        shot_type = 'Power'
                    elif direction_change > 0.5:  # High direction change
                        shot_type = 'Curved'
                    else:
                        shot_type = 'Regular'
                    
                    # Determine if shot is on target by projecting ball trajectory
                    # Get trajectory vector
                    traj_x = next_ball_x - curr_ball_x
                    traj_y = next_ball_y - curr_ball_y
                    
                    # Project to goal line
                    goal_x = target_goal[0]
                    dist_to_goal_x = abs(goal_x - curr_ball_x)
                    
                    if dist_to_goal_x > 0 and abs(traj_x) > 0:
                        # How many steps to reach goal line
                        scale_factor = dist_to_goal_x / abs(traj_x)
                        projected_y = curr_ball_y + scale_factor * traj_y
                        
                        # Check if projection hits goal
                        on_target = abs(projected_y - target_goal[1]) <= goal_width / 2
                    else:
                        on_target = False
                    
                    # Record the shot - use the actual frame number from the data
                    actual_frame = ball_df.index[i] if hasattr(ball_df.index, '__getitem__') else i
                    shots.append((closest_player, actual_frame, shot_type, on_target))
    
    return shots

# Enhanced version of detect_dribbles function
def detect_dribbles(ball_df, home_df, away_df, possession_radius=5.0, dribble_max_speed=7.0,
                 min_dribble_duration=3, min_dribble_distance=1.0):
    """
    Detect dribbles based on continuous ball possession by the same player.
    
    Enhanced version to leverage player influence model for better dribble detection.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame containing ball coordinates
    home_df, away_df : DataFrame
        DataFrames containing player coordinates
    possession_radius : float
        Maximum distance between player and ball to be in possession
    dribble_max_speed : float
        Maximum ball speed to be considered a dribble
    min_dribble_duration : int
        Minimum duration of a dribble in frames
    min_dribble_distance : float
        Minimum distance traveled during a dribble
    
    Returns:
    --------
    dribbles : list
        List of tuples (player_id, start_frame, end_frame, distance)
    """
    dribbles = []
    current_dribble = None
    
    # Calculate ball speeds if not already done
    if 'ball_speed' not in ball_df.columns and 'ball_velocity' not in ball_df.columns:
        ball_df['ball_speed'] = 0.0
        for i in range(1, len(ball_df)):
            dx = ball_df['ball_x'].iloc[i] - ball_df['ball_x'].iloc[i-1]
            dy = ball_df['ball_y'].iloc[i] - ball_df['ball_y'].iloc[i-1]
            ball_df.loc[ball_df.index[i], 'ball_speed'] = np.sqrt(dx**2 + dy**2)
    elif 'ball_velocity' in ball_df.columns and 'ball_speed' not in ball_df.columns:
        ball_df['ball_speed'] = ball_df['ball_velocity']
    
    # Detect continuous possession
    for i in range(len(ball_df)):
        # Skip if ball is moving too fast (likely a pass/shot)
        ball_speed = ball_df['ball_speed'].iloc[i] if 'ball_speed' in ball_df.columns else 0
        
        if ball_speed > dribble_max_speed:
            if current_dribble:
                # End current dribble
                player_id, start_frame, start_pos = current_dribble
                end_frame = i - 1
                end_pos = (ball_df['ball_x'].iloc[end_frame], ball_df['ball_y'].iloc[end_frame])
                
                # Calculate dribble distance
                dribble_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                
                # Only record dribbles that meet minimum criteria
                if (end_frame - start_frame >= min_dribble_duration and 
                    dribble_distance >= min_dribble_distance):
                    dribbles.append((player_id, start_frame, end_frame, dribble_distance))
                
                current_dribble = None
            continue
        
        # Get current ball position
        ball_pos = (ball_df['ball_x'].iloc[i], ball_df['ball_y'].iloc[i])
        
        # Find closest player to ball
        closest_player = None
        min_dist = float('inf')
        
        # Check home players
        for col in home_df.columns:
            if '_x' in col and 'home_' in col:
                player_id = col.replace('_x', '')
                y_col = f"{player_id}_y"
                if y_col in home_df.columns:
                    x = home_df[col].iloc[i]  # Direct indexing
                    y = home_df[y_col].iloc[i]
                    dist = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = player_id
        
        # Check away players
        for col in away_df.columns:
            if '_x' in col and 'away_' in col:
                player_id = col.replace('_x', '')
                y_col = f"{player_id}_y"
                if y_col in away_df.columns:
                    x = away_df[col].iloc[i]  # Direct indexing
                    y = away_df[y_col].iloc[i]
                    dist = np.sqrt((x - ball_pos[0])**2 + (y - ball_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = player_id
        
        # Only consider possession if player is within possession radius
        if closest_player and min_dist <= possession_radius:
            if current_dribble:
                player_id, start_frame, start_pos = current_dribble
                if player_id == closest_player:
                    # Continue current dribble
                    continue
                else:
                    # End current dribble and start a new one
                    end_frame = i - 1
                    end_pos = (ball_df['ball_x'].iloc[end_frame], ball_df['ball_y'].iloc[end_frame])
                    
                    # Calculate dribble distance
                    dribble_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                    
                    # Only record dribbles that meet minimum criteria
                    if (end_frame - start_frame >= min_dribble_duration and 
                        dribble_distance >= min_dribble_distance):
                        dribbles.append((player_id, start_frame, end_frame, dribble_distance))
            
            # Start new dribble
            current_dribble = (closest_player, i, ball_pos)
        elif current_dribble:
            # Lost possession
            player_id, start_frame, start_pos = current_dribble
            end_frame = i - 1
            end_pos = (ball_df['ball_x'].iloc[end_frame], ball_df['ball_y'].iloc[end_frame])
            
            # Calculate dribble distance
            dribble_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            # Only record dribbles that meet minimum criteria
            if (end_frame - start_frame >= min_dribble_duration and 
                dribble_distance >= min_dribble_distance):
                dribbles.append((player_id, start_frame, end_frame, dribble_distance))
            
            current_dribble = None
    
    # Handle the case where a dribble continues until the end of the data
    if current_dribble:
        player_id, start_frame, start_pos = current_dribble
        end_frame = len(ball_df) - 1
        end_pos = (ball_df['ball_x'].iloc[end_frame], ball_df['ball_y'].iloc[end_frame])
        
        # Calculate dribble distance
        dribble_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Only record dribbles that meet minimum criteria
        if (end_frame - start_frame >= min_dribble_duration and 
            dribble_distance >= min_dribble_distance):
            dribbles.append((player_id, start_frame, end_frame, dribble_distance))
    
    return dribbles

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive Movement Pattern Analysis')
    
    parser.add_argument('--match_num', type=int, default=0,
                        help='Match number to analyze (use match_num=4 for predicted data)')
    parser.add_argument('--predictions_file', type=str, default=None,
                        help='Path to prediction file (if using predicted ball coordinates)')
    parser.add_argument('--enhanced_model', action='store_true',
                        help='Use enhanced hybrid model with player influence features')
    parser.add_argument('--output_dir', type=str, default='analysis/movement_patterns',
                        help='Directory to save analysis results')
    parser.add_argument('--start_frame', type=int, default=1000,
                        help='Starting frame for analysis')
    parser.add_argument('--num_frames', type=int, default=10000,
                        help='Number of frames to analyze')
    parser.add_argument('--possession_radius', type=float, default=5.0,
                        help='Distance threshold for ball possession (meters)')
    parser.add_argument('--ball_speed_threshold', type=float, default=5.0,
                        help='Speed threshold for pass detection (meters/second)')
    parser.add_argument('--shot_speed_threshold', type=float, default=12.0,
                        help='Speed threshold for shot detection (meters/second)')
    parser.add_argument('--dribble_max_speed', type=float, default=7.0,
                        help='Maximum speed for dribble detection (meters/second)')
    
    return parser.parse_args()

def draw_soccer_field(ax, length=105, width=68):
    """Draw a soccer field on the given axes."""
    # Field outline
    rect = patches.Rectangle((0, 0), length, width, linewidth=2, 
                           edgecolor='white', facecolor='darkgreen', alpha=0.6)
    ax.add_patch(rect)
    
    # Halfway line
    ax.plot([length/2, length/2], [0, width], 'white', linewidth=2)
    
    # Center circle
    center_circle = patches.Circle((length/2, width/2), 9.15, 
                                 linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(center_circle)
    
    # Center dot
    center_dot = patches.Circle((length/2, width/2), 0.5, 
                              edgecolor='white', facecolor='white')
    ax.add_patch(center_dot)
    
    # Penalty areas
    # Left penalty area
    left_penalty = patches.Rectangle((0, width/2 - 20.16/2), 16.5, 20.16, 
                                    linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(left_penalty)
    
    # Right penalty area
    right_penalty = patches.Rectangle((length - 16.5, width/2 - 20.16/2), 16.5, 20.16, 
                                     linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(right_penalty)
    
    # Goal areas
    # Left goal area
    left_goal_area = patches.Rectangle((0, width/2 - 9.16/2), 5.5, 9.16, 
                                      linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(left_goal_area)
    
    # Right goal area
    right_goal_area = patches.Rectangle((length - 5.5, width/2 - 9.16/2), 5.5, 9.16, 
                                       linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(right_goal_area)
    
    # Goals
    goal_width = 7.32
    # Left goal
    left_goal = patches.Rectangle((-2, width/2 - goal_width/2), 2, goal_width, 
                                linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(left_goal)
    
    # Right goal
    right_goal = patches.Rectangle((length, width/2 - goal_width/2), 2, goal_width, 
                                  linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(right_goal)
    
    # Set axis limits
    ax.set_xlim(-5, length+5)
    ax.set_ylim(-5, width+5)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

def analyze_combined_patterns(ball_df, home_df, away_df, passes, shots, dribbles, args):
    """Create a combined heatmap analysis of all movement patterns."""
    print("Creating combined movement pattern heatmap...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create grid for heatmap
    field_length = 105
    field_width = 68
    grid_size = 1  # meters per grid cell
    x_bins = np.arange(0, field_length + grid_size, grid_size)
    y_bins = np.arange(0, field_width + grid_size, grid_size)
    
    # Initialize heatmaps for different movement types
    pass_heatmap = np.zeros((len(y_bins)-1, len(x_bins)-1))
    shot_heatmap = np.zeros((len(y_bins)-1, len(x_bins)-1))
    dribble_heatmap = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    # Instead of using complex frame indexing, let's simply use direct access by integer index
    # to avoid non-integer key errors
    
    # Populate pass heatmap
    for from_player, to_player, start_frame, end_frame in passes:
        # Find the corresponding indices in our filtered dataframe
        start_idx = frame_to_idx.get(start_frame, None)
        end_idx = frame_to_idx.get(end_frame, None)
        
        if start_idx is not None and end_idx is not None:
            if 0 <= start_idx < len(ball_df) and 0 <= end_idx < len(ball_df):
                try:
                    start_x = ball_df.iloc[start_idx]['ball_x']
                    start_y = ball_df.iloc[start_idx]['ball_y']
                    end_x = ball_df.iloc[end_idx]['ball_x']
                    end_y = ball_df.iloc[end_idx]['ball_y']
                except (KeyError, TypeError):
                    # Skip if we can't get the coordinates
                    continue
                
                # Add to heatmap using all points along the pass trajectory
                for t in np.linspace(0, 1, 10):
                    x = start_x + t * (end_x - start_x)
                    y = start_y + t * (end_y - start_y)
                    
                    if 0 <= x < field_length and 0 <= y < field_width:
                        x_idx = int(x / grid_size)
                        y_idx = int(y / grid_size)
                        pass_heatmap[y_idx, x_idx] += 1
    
    # Populate shot heatmap
    for player_id, frame, shot_type, on_target in shots:
        # Find the corresponding index in our filtered dataframe
        frame_idx = frame_to_idx.get(frame, None)
        
        if frame_idx is not None and 0 <= frame_idx < len(ball_df):
            try:
                x = ball_df.iloc[frame_idx]['ball_x']
                y = ball_df.iloc[frame_idx]['ball_y']
                
                # Add to heatmap
                x_bin = min(max(0, int(x / grid_size)), len(x_bins) - 2)
                y_bin = min(max(0, int(y / grid_size)), len(y_bins) - 2)
                shot_heatmap[y_bin, x_bin] += 1
            except (KeyError, TypeError):
                # Skip if we can't get the coordinates
                continue
    
    # Populate dribble heatmap
    for player_id, start_frame, end_frame, distance in dribbles:
        # Find the corresponding indices in our filtered dataframe
        start_idx = frame_to_idx.get(start_frame, None)
        end_idx = frame_to_idx.get(end_frame, None)
        
        if start_idx is not None and end_idx is not None:
            if 0 <= start_idx < len(ball_df) and 0 <= end_idx < len(ball_df):
                # Get all frame indices between start and end
                frame_indices = []
                for i in range(len(ball_df)):
                    frame = ball_df.index[i] if hasattr(ball_df.index, '__getitem__') else i
                    if start_idx <= i <= end_idx:
                        frame_indices.append(i)
                
                for frame_idx in frame_indices:
                    try:
                        x = ball_df.iloc[frame_idx]['ball_x']
                        y = ball_df.iloc[frame_idx]['ball_y']
                        
                        # Add to heatmap
                        x_bin = min(max(0, int(x / grid_size)), len(x_bins) - 2)
                        y_bin = min(max(0, int(y / grid_size)), len(y_bins) - 2)
                        dribble_heatmap[y_bin, x_bin] += 1
                    except (KeyError, TypeError, IndexError):
                        # Skip if we can't get the coordinates
                        continue
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Field with passes
    ax = axes[0, 0]
    draw_soccer_field(ax)
    # Plot passes
    for from_player, to_player, start_frame, end_frame in passes:
        if start_frame < len(ball_df) and end_frame < len(ball_df):
            base_frame = ball_df.iloc[0]['frame']
            start_idx = start_frame - base_frame if start_frame >= base_frame else 0
            end_idx = end_frame - base_frame if end_frame >= base_frame else 0
            
            if 0 <= start_idx < len(ball_df) and 0 <= end_idx < len(ball_df):
                start_x = ball_df.iloc[start_idx]['ball_x']
                start_y = ball_df.iloc[start_idx]['ball_y']
                end_x = ball_df.iloc[end_idx]['ball_x']
                end_y = ball_df.iloc[end_idx]['ball_y']
                
                # Determine team color
                color = 'blue' if 'home_' in from_player else 'red'
                alpha = 0.3
                
                # Draw pass arrow
                ax.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                       color=color, alpha=alpha, width=0.2, head_width=1.0, length_includes_head=True)
    ax.set_title('Pass Map', fontsize=14)
    
    # Field with shots
    ax = axes[0, 1]
    draw_soccer_field(ax)
    # Plot shots
    for player_id, frame, shot_type, on_target in shots:
        if frame < len(ball_df):
            base_frame = ball_df.iloc[0]['frame']
            frame_idx = frame - base_frame if frame >= base_frame else 0
            
            if 0 <= frame_idx < len(ball_df):
                x = ball_df.iloc[frame_idx]['ball_x']
                y = ball_df.iloc[frame_idx]['ball_y']
                
                # Determine color and marker style
                color = 'lime' if on_target else 'orange'
                marker = '*'
                markersize = 300
                
                # Draw shot line to goal
                if player_id.startswith('home_'):
                    # Home shoots right-to-left
                    goal_x, goal_y = 105, 34
                else:
                    # Away shoots left-to-right
                    goal_x, goal_y = 0, 34
                
                ax.plot([x, goal_x], [y, goal_y], '--', color=color, alpha=0.5, linewidth=1.5)
                
                # Draw shot origin
                ax.scatter(x, y, s=markersize, marker=marker, color=color, edgecolors='black', zorder=3)
    ax.set_title('Shot Map', fontsize=14)
    
    # Field with dribbles
    ax = axes[1, 0]
    draw_soccer_field(ax)
    # Plot dribbles
    for player_id, start_frame, end_frame, distance in dribbles:
        if start_frame < len(ball_df) and end_frame < len(ball_df):
            base_frame = ball_df.iloc[0]['frame']
            start_idx = start_frame - base_frame if start_frame >= base_frame else 0
            end_idx = end_frame - base_frame if end_frame >= base_frame else 0
            
            if 0 <= start_idx < len(ball_df) and 0 <= end_idx < len(ball_df) and end_idx - start_idx > 0:
                # Get ball positions during dribble
                dribble_positions_x = ball_df.iloc[start_idx:end_idx+1]['ball_x'].values
                dribble_positions_y = ball_df.iloc[start_idx:end_idx+1]['ball_y'].values
                
                # Determine team color
                color = 'blue' if 'home_' in player_id else 'red'
                
                # Draw dribbling path
                ax.plot(dribble_positions_x, dribble_positions_y, color=color, linewidth=2, alpha=0.7)
                
                # Mark start and end points
                ax.scatter(dribble_positions_x[0], dribble_positions_y[0], color=color, s=30, marker='o')
                ax.scatter(dribble_positions_x[-1], dribble_positions_y[-1], color=color, s=50, marker='s')
    ax.set_title('Dribble Map', fontsize=14)
    
    # Combined heatmap of all movement types
    ax = axes[1, 1]
    combined_heatmap = pass_heatmap + shot_heatmap + dribble_heatmap
    draw_soccer_field(ax)
    heatmap = ax.imshow(combined_heatmap, cmap='hot', origin='lower', 
                       extent=[0, field_length, 0, field_width], alpha=0.7)
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label('Activity Intensity')
    ax.set_title('Combined Movement Heatmap', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'combined_movement_analysis_match_{args.match_num}.png'), dpi=300)
    plt.close()
    
    # Team-specific movement analysis
    # Split passes by team
    home_passes = [p for p in passes if p[0].startswith('home_') and p[1].startswith('home_')]
    away_passes = [p for p in passes if p[0].startswith('away_') and p[1].startswith('away_')]
    
    # Split shots by team
    home_shots = [s for s in shots if s[0].startswith('home_')]
    away_shots = [s for s in shots if s[0].startswith('away_')]
    
    # Split dribbles by team
    home_dribbles = [d for d in dribbles if d[0].startswith('home_')]
    away_dribbles = [d for d in dribbles if d[0].startswith('away_')]
    
    # Create comparison metrics
    comparison_data = {
        'Metric': [
            'Total Passes', 'Pass Success Rate', 'Total Shots', 
            'Shot Accuracy', 'Total Dribbles', 'Avg Dribble Distance'
        ],
        'Home Team': [
            len(home_passes),
            f"{len([p for p in home_passes if p[3] - p[2] < 15])/len(home_passes)*100:.1f}%" if home_passes else "N/A",
            len(home_shots),
            f"{len([s for s in home_shots if s[3]])/len(home_shots)*100:.1f}%" if home_shots else "N/A",
            len(home_dribbles),
            f"{sum([d[3] for d in home_dribbles])/len(home_dribbles):.2f}m" if home_dribbles else "N/A"
        ],
        'Away Team': [
            len(away_passes),
            f"{len([p for p in away_passes if p[3] - p[2] < 15])/len(away_passes)*100:.1f}%" if away_passes else "N/A",
            len(away_shots),
            f"{len([s for s in away_shots if s[3]])/len(away_shots)*100:.1f}%" if away_shots else "N/A",
            len(away_dribbles),
            f"{sum([d[3] for d in away_dribbles])/len(away_dribbles):.2f}m" if away_dribbles else "N/A"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nTeam Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(args.output_dir, f'team_comparison_match_{args.match_num}.csv'), index=False)
    
    return comparison_df

def calculate_player_influence_metrics(ball_df, home_df, away_df, passes, shots, dribbles):
    """Calculate player influence metrics based on all movement patterns."""
    # Get unique players from all patterns
    all_players = set()
    for p in passes:
        all_players.add(p[0])
        all_players.add(p[1])
    for s in shots:
        all_players.add(s[0])
    for d in dribbles:
        all_players.add(d[0])
    
    player_metrics = {}
    for player in all_players:
        # Initialize metrics
        player_metrics[player] = {
            'passes_made': 0,
            'passes_received': 0,
            'shots': 0,
            'shots_on_target': 0,
            'dribbles': 0,
            'dribble_distance': 0,
            'possession_time': 0  # in frames
        }
    
    # Count passes
    for from_player, to_player, start_frame, end_frame in passes:
        if from_player in player_metrics:
            player_metrics[from_player]['passes_made'] += 1
        if to_player in player_metrics:
            player_metrics[to_player]['passes_received'] += 1
    
    # Count shots
    for player_id, frame, shot_type, on_target in shots:
        if player_id in player_metrics:
            player_metrics[player_id]['shots'] += 1
            if on_target:
                player_metrics[player_id]['shots_on_target'] += 1
    
    # Count dribbles
    for player_id, start_frame, end_frame, distance in dribbles:
        if player_id in player_metrics:
            player_metrics[player_id]['dribbles'] += 1
            player_metrics[player_id]['dribble_distance'] += distance
            player_metrics[player_id]['possession_time'] += (end_frame - start_frame)
    
    # Calculate influence score
    for player, metrics in player_metrics.items():
        # Simple weighted score
        influence_score = (
            metrics['passes_made'] * 1.0 +
            metrics['passes_received'] * 0.8 +
            metrics['shots'] * 2.0 +
            metrics['shots_on_target'] * 3.0 +
            metrics['dribbles'] * 1.5 +
            metrics['dribble_distance'] * 0.1 +
            metrics['possession_time'] * 0.01
        )
        metrics['influence_score'] = influence_score
    
    # Convert to DataFrame
    player_df = pd.DataFrame.from_dict(player_metrics, orient='index')
    player_df['player_id'] = player_df.index
    player_df['team'] = player_df['player_id'].apply(lambda x: 'Home' if x.startswith('home_') else 'Away')
    player_df['player_number'] = player_df['player_id'].apply(lambda x: x.split('_')[1])
    
    # Sort by influence score
    player_df = player_df.sort_values('influence_score', ascending=False)
    
    return player_df

def main():
    """Main function to run the comprehensive movement pattern analysis."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing movement patterns for match {args.match_num}...")
    
    # Load data
    if args.match_num == 4 and args.predictions_file:
        # Using predicted ball coordinates for match 4
        print(f"Loading player data for match {args.match_num}...")
        _, home_df, away_df = load_data(args.match_num)
        
        print(f"Loading ball predictions from {args.predictions_file}...")
        predictions = pd.read_csv(args.predictions_file)
        
        # Create enhanced ball DataFrame from predictions
        ball_df = create_enhanced_ball_df(predictions, args.start_frame, args.num_frames)
        prediction_mode = True
    else:
        # Using actual ball coordinates
        print(f"Loading match {args.match_num} data...")
        ball_df, home_df, away_df = load_data(args.match_num)
        
        # Calculate ball velocity and other features
        print("Calculating ball velocity...")
        ball_df = calculate_ball_velocity(ball_df)
    
    # Get the appropriate frame range for analysis
    if 'frame' not in ball_df.columns:
        ball_df['frame'] = ball_df['Time']
    
    # Ensure we have the proper frame range
    min_frame = ball_df['frame'].min()
    max_frame = ball_df['frame'].max()
    start_frame = max(min_frame, args.start_frame)
    end_frame = min(max_frame, start_frame + args.num_frames)
    
    # Adjust frame range if too small
    if end_frame - start_frame < 1000:
        print(f"Warning: Selected frame range ({start_frame}-{end_frame}) is too small. Using all available frames.")
        start_frame = min_frame
        end_frame = max_frame
    
    print(f"Analyzing frames {start_frame} to {end_frame} (total: {end_frame - start_frame})")
    
    # Select relevant frames
    if 'Time' in ball_df.columns:
        frame_filter = (ball_df['Time'] >= start_frame) & (ball_df['Time'] < end_frame)
    else:
        frame_filter = (ball_df['frame'] >= start_frame) & (ball_df['frame'] < end_frame)
    
    # Get the filtered data
    ball_df_filtered = ball_df.loc[frame_filter].copy().reset_index(drop=True)
    
    # Make sure we have the correct indexes for all dataframes
    if 'Time' in home_df.columns and 'Time' in away_df.columns:
        home_filter = (home_df['Time'] >= start_frame) & (home_df['Time'] < end_frame)
        away_filter = (away_df['Time'] >= start_frame) & (away_df['Time'] < end_frame)
        home_df_filtered = home_df.loc[home_filter].copy().reset_index(drop=True)
        away_df_filtered = away_df.loc[away_filter].copy().reset_index(drop=True)
    else:
        # In case there's no Time column, try to use matching indexes
        idx_range = ball_df_filtered.index
        home_df_filtered = home_df.iloc[idx_range].copy().reset_index(drop=True)
        away_df_filtered = away_df.iloc[idx_range].copy().reset_index(drop=True)
    
    # Use the enhanced model if requested
    prediction_mode = args.match_num == 4 or args.enhanced_model
    
    # Show some data info
    print(f"Ball data shape: {ball_df_filtered.shape}")
    print(f"Home data shape: {home_df_filtered.shape}")
    print(f"Away data shape: {away_df_filtered.shape}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Detect Passes
    print("Detecting passes...")
    passes = detect_passes(
        ball_df_filtered, home_df_filtered, away_df_filtered,
        possession_radius=args.possession_radius,
        ball_speed_threshold=args.ball_speed_threshold,
        prediction_mode=prediction_mode
    )
    
    # 2. Detect Shots
    print("Detecting shots...")
    shots = detect_shots(
        ball_df_filtered, home_df_filtered, away_df_filtered,
        shot_speed_threshold=args.shot_speed_threshold,
        possession_radius=args.possession_radius
    )
    
    # 3. Detect Dribbles
    print("Detecting dribbles...")
    dribbles = detect_dribbles(
        ball_df_filtered, home_df_filtered, away_df_filtered,
        possession_radius=args.possession_radius,
        dribble_max_speed=args.dribble_max_speed
    )
    
    # Print summary statistics
    print("\nMovement Pattern Analysis Summary:")
    print(f"Total Passes: {len(passes)}")
    home_passes = len([p for p in passes if p[0].startswith('home_') and p[1].startswith('home_')])
    away_passes = len([p for p in passes if p[0].startswith('away_') and p[1].startswith('away_')])
    cross_team = len(passes) - home_passes - away_passes
    print(f"  Home Team Passes: {home_passes}")
    print(f"  Away Team Passes: {away_passes}")
    print(f"  Cross-Team Interactions: {cross_team}")
    
    print(f"Total Shots: {len(shots)}")
    on_target = len([s for s in shots if s[3]])
    off_target = len([s for s in shots if not s[3]])
    print(f"  On Target: {on_target}")
    print(f"  Off Target: {off_target}")
    print(f"  Shot Accuracy: {on_target/len(shots)*100:.1f}%" if shots else "  Shot Accuracy: N/A")
    
    print(f"Total Dribbles: {len(dribbles)}")
    home_dribbles = len([d for d in dribbles if d[0].startswith('home_')])
    away_dribbles = len([d for d in dribbles if d[0].startswith('away_')])
    print(f"  Home Team Dribbles: {home_dribbles}")
    print(f"  Away Team Dribbles: {away_dribbles}")
    
    # 4. Combined Analysis
    comparison_df = analyze_combined_patterns(ball_df_filtered, home_df_filtered, away_df_filtered, passes, shots, dribbles, args)
    
    # 5. Player influence analysis
    print("\nCalculating player influence metrics...")
    try:
        player_metrics = calculate_player_influence_metrics(ball_df_filtered, home_df_filtered, away_df_filtered, passes, shots, dribbles)
        
        # Print player metrics
        print("\nTop 5 Most Influential Players:")
        columns_to_display = [col for col in ['team', 'player_number', 'influence_score', 'passes_made', 
                         'shots', 'shots_on_target', 'dribbles'] if col in player_metrics.columns]
        
        if columns_to_display and not player_metrics.empty:
            print(player_metrics[columns_to_display].head(5).to_string(index=False))
        else:
            print("No player metrics available to display")
        
        # Save player metrics
        player_metrics.to_csv(os.path.join(args.output_dir, f'player_metrics_match_{args.match_num}.csv'), index=False)
    except Exception as e:
        print(f"\nError calculating player metrics: {str(e)}")
        print("Continuing with analysis...")
        player_metrics = pd.DataFrame()
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")
    
    return ball_df, passes, shots, dribbles, player_metrics

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
