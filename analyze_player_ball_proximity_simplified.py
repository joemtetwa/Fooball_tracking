import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from collections import defaultdict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import load_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Player Proximity to Predicted Ball')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_enhanced_predictions.csv',
                        help='Path to the enhanced predictions CSV file')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to analyze')
    parser.add_argument('--output_dir', type=str, default='analysis/player_ball_proximity',
                        help='Directory to save analysis results')
    parser.add_argument('--possession_threshold', type=float, default=7.0,
                        help='Distance threshold for ball possession (meters)')
    parser.add_argument('--start_frame', type=int, default=1000,
                        help='Starting frame for analysis')
    parser.add_argument('--num_frames', type=int, default=5000,
                        help='Number of frames to analyze')
    
    return parser.parse_args()

def calculate_player_distances(ball_pos, player_positions):
    """Calculate distances from all players to the ball."""
    distances = []
    
    for player_id, x, y, is_home in player_positions:
        if pd.isna(x) or pd.isna(y):
            continue
            
        dist = np.sqrt((ball_pos[0] - x)**2 + (ball_pos[1] - y)**2)
        distances.append((player_id, dist, is_home))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    
    return distances

def determine_possession(distances, possession_threshold):
    """Determine which player has possession of the ball."""
    if not distances:
        return None, False, float('inf')
    
    # Get closest player
    closest_player, min_distance, is_home = distances[0]
    
    # Check if within possession threshold
    if min_distance <= possession_threshold:
        return closest_player, is_home, min_distance
    else:
        return None, False, min_distance

def analyze_player_proximity(predictions, home_df, away_df, args):
    """Analyze proximity of players to the predicted ball coordinates."""
    print("Analyzing player proximity to predicted ball coordinates...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Initialize arrays to store results
    frame_range = range(args.start_frame, min(args.start_frame + args.num_frames, len(predictions)))
    num_frames = len(frame_range)
    
    possession_data = []
    closest_players_data = []
    player_possession_count = {}
    player_proximity_count = defaultdict(int)  # Count times when player is within 2x threshold
    team_possession_frames = {'home': 0, 'away': 0, 'none': 0}
    
    # Scale factor for coordinates (predictions are in different scale than player coordinates)
    # This scaling factor should align ball coordinates with player coordinates
    ball_scale_factor = 100.0
    
    # Analyze each frame
    for i, frame_idx in enumerate(frame_range):
        if i % 500 == 0:
            print(f"Processing frame {frame_idx} ({i+1}/{num_frames})...")
        
        # Get ball position
        if frame_idx < len(predictions):
            ball_x = predictions.iloc[frame_idx]['ball_x'] / ball_scale_factor
            ball_y = predictions.iloc[frame_idx]['ball_y'] / ball_scale_factor
            ball_pos = (ball_x, ball_y)
        else:
            continue
        
        # Get player positions for this frame
        player_positions = []
        
        # Add home player positions
        for x_col, y_col in zip(home_x_cols, home_y_cols):
            if frame_idx < len(home_df):
                x = home_df.iloc[frame_idx][x_col]
                y = home_df.iloc[frame_idx][y_col]
                player_id = x_col.replace('_x', '')
                player_positions.append((player_id, x, y, True))  # True = home team
        
        # Add away player positions
        for x_col, y_col in zip(away_x_cols, away_y_cols):
            if frame_idx < len(away_df):
                x = away_df.iloc[frame_idx][x_col]
                y = away_df.iloc[frame_idx][y_col]
                player_id = x_col.replace('_x', '')
                player_positions.append((player_id, x, y, False))  # False = away team
        
        # Calculate distances from ball to each player
        distances = calculate_player_distances(ball_pos, player_positions)
        
        # Count players in proximity (within 2x possession threshold)
        for player_id, dist, is_home in distances:
            if dist <= args.possession_threshold * 2:
                player_proximity_count[player_id] += 1
        
        # Determine possession
        player_with_possession, is_home, min_distance = determine_possession(
            distances, args.possession_threshold
        )
        
        # Update possession data
        if player_with_possession:
            team = 'home' if is_home else 'away'
            team_possession_frames[team] += 1
            
            if player_with_possession in player_possession_count:
                player_possession_count[player_with_possession] += 1
            else:
                player_possession_count[player_with_possession] = 1
        else:
            team_possession_frames['none'] += 1
        
        # Save data for this frame
        possession_data.append({
            'frame': frame_idx,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'player_with_possession': player_with_possession,
            'team_with_possession': 'home' if is_home and player_with_possession else 'away' if player_with_possession else 'none',
            'min_distance': min_distance
        })
        
        # Save top 5 closest players
        for j, (player_id, dist, is_home) in enumerate(distances[:5]):
            if j == 0 or dist <= args.possession_threshold * 2:  # Only include players within 2x possession threshold
                closest_players_data.append({
                    'frame': frame_idx,
                    'player_id': player_id,
                    'distance': dist,
                    'team': 'home' if is_home else 'away',
                    'rank': j + 1
                })
    
    # Convert to DataFrames
    possession_df = pd.DataFrame(possession_data)
    closest_players_df = pd.DataFrame(closest_players_data)
    
    # Calculate team possession percentages
    total_frames = num_frames
    home_pct = team_possession_frames['home'] / total_frames * 100
    away_pct = team_possession_frames['away'] / total_frames * 100
    no_possession_pct = team_possession_frames['none'] / total_frames * 100
    
    print("\nTeam Possession Analysis:")
    print(f"Home Team: {home_pct:.1f}% ({team_possession_frames['home']} frames)")
    print(f"Away Team: {away_pct:.1f}% ({team_possession_frames['away']} frames)")
    print(f"No Clear Possession: {no_possession_pct:.1f}% ({team_possession_frames['none']} frames)")
    
    # Print top players by possession
    print("\nTop Players by Possession Time:")
    player_possession = sorted(player_possession_count.items(), key=lambda x: x[1], reverse=True)
    for player_id, count in player_possession[:10]:  # Top 10 players
        team = 'Home' if 'home_' in player_id else 'Away'
        pct = count / total_frames * 100
        print(f"{player_id} ({team}): {pct:.1f}% ({count} frames)")
    
    # Print top players by proximity (being close to the ball)
    print("\nTop Players by Proximity to Ball:")
    player_proximity = sorted(player_proximity_count.items(), key=lambda x: x[1], reverse=True)
    for player_id, count in player_proximity[:10]:  # Top 10 players
        team = 'Home' if 'home_' in player_id else 'Away'
        pct = count / total_frames * 100
        print(f"{player_id} ({team}): {pct:.1f}% ({count} frames)")
    
    # Save DataFrames to CSV
    possession_df.to_csv(os.path.join(args.output_dir, 'possession_analysis.csv'), index=False)
    closest_players_df.to_csv(os.path.join(args.output_dir, 'closest_players_analysis.csv'), index=False)
    
    # Create visualizations
    create_visualizations(possession_df, closest_players_df, player_possession_count, 
                         player_proximity_count, team_possession_frames, args)
    
    return possession_df, closest_players_df

def create_visualizations(possession_df, closest_players_df, player_possession_count, 
                        player_proximity_count, team_possession_frames, args):
    """Create visualizations for player proximity analysis."""
    print("Creating visualizations...")
    
    # 1. Team Possession Pie Chart
    plt.figure(figsize=(8, 6))
    labels = ['Home Team', 'Away Team', 'No Clear Possession']
    sizes = [team_possession_frames['home'], team_possession_frames['away'], team_possession_frames['none']]
    colors = ['blue', 'red', 'gray']
    
    # Only include non-zero values
    non_zero_labels = []
    non_zero_sizes = []
    non_zero_colors = []
    
    for i, size in enumerate(sizes):
        if size > 0:
            non_zero_labels.append(labels[i])
            non_zero_sizes.append(size)
            non_zero_colors.append(colors[i])
    
    if non_zero_sizes:
        plt.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Ball Possession Distribution (Frames {args.start_frame}-{args.start_frame+args.num_frames-1})')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'team_possession_pie_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top Players by Possession Bar Chart
    if player_possession_count:
        plt.figure(figsize=(10, 6))
        player_possession = sorted(player_possession_count.items(), key=lambda x: x[1], reverse=True)[:15]  # Top 15 players
        players = [p[0].replace('home_', 'H').replace('away_', 'A') for p in player_possession]
        counts = [p[1] for p in player_possession]
        colors = ['blue' if 'home_' in p[0] else 'red' for p in player_possession]
        
        plt.bar(players, counts, color=colors)
        plt.xlabel('Player ID')
        plt.ylabel('Frames with Possession')
        plt.title('Top Players by Ball Possession')
        plt.xticks(rotation=45, ha='right')
        
        # Add team labels in legend
        if 'blue' in colors:
            home_patch = plt.Rectangle((0, 0), 1, 1, color='blue', label='Home Team')
            plt.legend(handles=[home_patch])
        
        if 'red' in colors:
            away_patch = plt.Rectangle((0, 0), 1, 1, color='red', label='Away Team')
            if 'blue' in colors:
                plt.legend(handles=[home_patch, away_patch])
            else:
                plt.legend(handles=[away_patch])
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'top_players_possession.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Players by Proximity Bar Chart
    plt.figure(figsize=(10, 6))
    player_proximity = sorted(player_proximity_count.items(), key=lambda x: x[1], reverse=True)[:15]  # Top 15 players
    players = [p[0].replace('home_', 'H').replace('away_', 'A') for p in player_proximity]
    counts = [p[1] for p in player_proximity]
    colors = ['blue' if 'home_' in p[0] else 'red' for p in player_proximity]
    
    plt.bar(players, counts, color=colors)
    plt.xlabel('Player ID')
    plt.ylabel(f'Frames within {args.possession_threshold * 2}m of Ball')
    plt.title('Top Players by Proximity to Ball')
    plt.xticks(rotation=45, ha='right')
    
    # Add team labels in legend
    home_patch = plt.Rectangle((0, 0), 1, 1, color='blue', label='Home Team')
    away_patch = plt.Rectangle((0, 0), 1, 1, color='red', label='Away Team')
    plt.legend(handles=[home_patch, away_patch])
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top_players_proximity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distance to Ball Over Time for select players
    plt.figure(figsize=(12, 6))
    
    # Get top 5 players by proximity
    top_players = [p[0] for p in player_proximity[:5]]
    
    # Filter data for top players
    for player_id in top_players:
        player_data = closest_players_df[closest_players_df['player_id'] == player_id]
        if not player_data.empty:
            team = 'Home' if 'home_' in player_id else 'Away'
            color = 'blue' if team == 'Home' else 'red'
            
            # Only plot every 20th point to reduce density
            sampled_data = player_data.iloc[::20]
            plt.plot(sampled_data['frame'], sampled_data['distance'], 
                     label=f"{player_id} ({team})", color=color, alpha=0.7)
    
    # Add possession threshold line
    plt.axhline(y=args.possession_threshold, color='k', linestyle='--', alpha=0.7, 
                label=f'Possession Threshold ({args.possession_threshold} m)')
    
    plt.xlabel('Frame')
    plt.ylabel('Distance to Ball (meters)')
    plt.title('Distance to Ball Over Time for Top Players')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'distance_to_ball_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Heat map of ball positions by possession
    create_ball_position_heatmap(possession_df, args)

def create_ball_position_heatmap(possession_df, args):
    """Create a heatmap of ball positions colored by team possession."""
    plt.figure(figsize=(10, 7))
    
    # Set field dimensions in meters
    field_length = 105
    field_width = 68
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Separate by team possession
    home_possession = possession_df[possession_df['team_with_possession'] == 'home']
    away_possession = possession_df[possession_df['team_with_possession'] == 'away']
    no_possession = possession_df[possession_df['team_with_possession'] == 'none']
    
    # Plot points, colored by team possession
    if not home_possession.empty:
        plt.scatter(home_possession['ball_x'], home_possession['ball_y'], 
                   c='blue', alpha=0.6, s=10, label='Home Team Possession')
    
    if not away_possession.empty:
        plt.scatter(away_possession['ball_x'], away_possession['ball_y'], 
                   c='red', alpha=0.6, s=10, label='Away Team Possession')
    
    plt.scatter(no_possession['ball_x'], no_possession['ball_y'], 
               c='gray', alpha=0.3, s=5, label='No Clear Possession')
    
    plt.title('Ball Position Heatmap by Team Possession')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(args.output_dir, 'ball_position_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Analyze player proximity
        possession_df, closest_players_df = analyze_player_proximity(predictions, home_df, away_df, args)
        print("Analysis complete. Results saved to:", args.output_dir)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save partial results
        if 'possession_df' in locals() and not possession_df.empty:
            possession_df.to_csv(os.path.join(args.output_dir, 'possession_analysis_partial.csv'), index=False)
            print("Saved partial possession analysis results.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
