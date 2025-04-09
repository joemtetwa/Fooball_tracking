import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import from existing modules
from src.hybrid_lstm_gnn.data_utils import load_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create Pass Network Visualizations')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_enhanced_predictions.csv',
                        help='Path to the enhanced predictions CSV file')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to analyze')
    parser.add_argument('--output_dir', type=str, default='visualizations/pass_networks',
                        help='Directory to save visualizations')
    parser.add_argument('--possession_threshold', type=float, default=2.0,
                        help='Distance threshold for ball possession (meters)')
    parser.add_argument('--min_passes', type=int, default=3,
                        help='Minimum number of passes to include in the network')
    
    return parser.parse_args()

def determine_possession(player_positions, ball_pos, possession_threshold=2.0):
    """
    Determine which player has possession of the ball.
    
    Args:
        player_positions: List of player positions [(x1, y1, player_id, is_home), ...]
        ball_pos: Ball position (x, y)
        possession_threshold: Distance threshold for ball possession
        
    Returns:
        player_idx: Index of player with possession, or None if no player has possession
        is_home: True if home team has possession, False if away team
        player_id: ID of player with possession
    """
    if len(player_positions) == 0:
        return None, False, None
    
    # Calculate distances from ball to each player
    distances = []
    for pos in player_positions:
        player_pos = np.array([pos[0], pos[1]])
        ball_pos_array = np.array(ball_pos)
        distance = np.linalg.norm(player_pos - ball_pos_array)
        distances.append(distance)
    
    # Find the closest player
    closest_idx = np.argmin(distances)
    min_distance = distances[closest_idx]
    
    # Check if the closest player is within the possession threshold
    if min_distance <= possession_threshold:
        # Get player information
        player_id = player_positions[closest_idx][2]
        is_home = player_positions[closest_idx][3]
        return closest_idx, is_home, player_id
    else:
        return None, False, None

def detect_passes(ball_predictions, home_df, away_df, possession_threshold=2.0, min_frames_between_passes=10):
    """
    Detect passes between players based on ball movement and player positions.
    
    Args:
        ball_predictions: DataFrame with predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        possession_threshold: Distance threshold for ball possession
        min_frames_between_passes: Minimum number of frames between consecutive passes
        
    Returns:
        passes: List of detected passes [(from_player, to_player, from_is_home, to_is_home, frame), ...]
    """
    print("Detecting passes based on predicted ball coordinates...")
    
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
    
    # Initialize variables for pass detection
    passes = []
    current_possession = None
    last_possession_change = 0
    
    # Process each frame
    for frame in range(len(ball_predictions)):
        # Get ball position (scale to meters)
        ball_x = ball_predictions.iloc[frame]['ball_x'] / 10.0
        ball_y = ball_predictions.iloc[frame]['ball_y'] / 10.0
        ball_pos = (ball_x, ball_y)
        
        # Get player positions for this frame
        player_positions = []
        
        # Add home player positions
        for x_col, y_col in zip(home_x_cols, home_y_cols):
            if frame < len(home_df):
                x = home_df.iloc[frame][x_col]
                y = home_df.iloc[frame][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    player_id = x_col.replace('_x', '')
                    player_positions.append((x, y, player_id, True))  # True = home team
        
        # Add away player positions
        for x_col, y_col in zip(away_x_cols, away_y_cols):
            if frame < len(away_df):
                x = away_df.iloc[frame][x_col]
                y = away_df.iloc[frame][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    player_id = x_col.replace('_x', '')
                    player_positions.append((x, y, player_id, False))  # False = away team
        
        # Determine possession
        player_idx, is_home, player_id = determine_possession(
            player_positions, ball_pos, possession_threshold
        )
        
        # Check for possession change
        if player_id is not None:
            if current_possession is None:
                # First possession
                current_possession = (player_id, is_home)
            elif player_id != current_possession[0]:
                # Possession changed to a different player
                if frame - last_possession_change >= min_frames_between_passes:
                    # Record pass
                    from_player, from_is_home = current_possession
                    to_player, to_is_home = player_id, is_home
                    passes.append((from_player, to_player, from_is_home, to_is_home, frame))
                    
                    # Update possession
                    current_possession = (player_id, is_home)
                    last_possession_change = frame
    
    return passes

def create_pass_network(passes, team_prefix, min_passes=3):
    """
    Create a pass network for a team based on detected passes.
    
    Args:
        passes: List of detected passes
        team_prefix: Prefix for team players ('home_' or 'away_')
        min_passes: Minimum number of passes to include in the network
        
    Returns:
        G: NetworkX graph representing the pass network
        pos: Dictionary of node positions
        weights: Dictionary of edge weights
    """
    # Filter passes for the specified team
    is_home = team_prefix == 'home_'
    team_passes = [p for p in passes if (p[2] == is_home and p[3] == is_home)]
    
    # Count passes between players
    pass_counts = {}
    for from_player, to_player, _, _, _ in team_passes:
        if (from_player, to_player) in pass_counts:
            pass_counts[(from_player, to_player)] += 1
        else:
            pass_counts[(from_player, to_player)] = 1
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes (players)
    player_ids = set()
    for from_player, to_player, _, _, _ in team_passes:
        player_ids.add(from_player)
        player_ids.add(to_player)
    
    for player_id in player_ids:
        G.add_node(player_id)
    
    # Add edges (passes)
    weights = {}
    for (from_player, to_player), count in pass_counts.items():
        if count >= min_passes:
            G.add_edge(from_player, to_player, weight=count)
            weights[(from_player, to_player)] = count
    
    return G, weights

def get_average_player_positions(home_df, away_df, team_prefix):
    """
    Calculate average positions for players.
    
    Args:
        home_df, away_df: DataFrames with player positions
        team_prefix: Prefix for team players ('home_' or 'away_')
        
    Returns:
        positions: Dictionary of average player positions {player_id: (x, y)}
    """
    df = home_df if team_prefix == 'home_' else away_df
    
    # Get player position columns
    x_cols = [col for col in df.columns if '_x' in col and team_prefix in col]
    y_cols = [col for col in df.columns if '_y' in col and team_prefix in col]
    
    # Sort columns to ensure consistent order
    x_cols.sort()
    y_cols.sort()
    
    # Calculate average positions
    positions = {}
    for x_col, y_col in zip(x_cols, y_cols):
        player_id = x_col.replace('_x', '')
        x_values = df[x_col].dropna().values
        y_values = df[y_col].dropna().values
        
        if len(x_values) > 0 and len(y_values) > 0:
            avg_x = np.mean(x_values)
            avg_y = np.mean(y_values)
            positions[player_id] = (avg_x, avg_y)
    
    return positions

def visualize_pass_network(G, positions, weights, team_name, output_path, field_length=105, field_width=68):
    """
    Visualize pass network for a team.
    
    Args:
        G: NetworkX graph representing the pass network
        positions: Dictionary of node positions
        weights: Dictionary of edge weights
        team_name: Name of the team ('Home' or 'Away')
        output_path: Path to save the visualization
        field_length, field_width: Field dimensions in meters
    """
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
    
    # Set node color based on team
    node_color = 'blue' if team_name == 'Home' else 'red'
    
    # Calculate node sizes based on degree centrality
    degree = dict(G.degree())
    node_sizes = {node: 300 + 100 * degree.get(node, 0) for node in G.nodes()}
    
    # Draw nodes (players)
    for node in G.nodes():
        if node in positions:
            x, y = positions[node]
            size = node_sizes[node]
            plt.scatter(x, y, s=size, c=node_color, alpha=0.7, edgecolors='black', zorder=5)
            
            # Add player label
            label = node.replace('home_', '').replace('away_', '')
            plt.text(x, y + 1, label, fontsize=10, ha='center', va='center', zorder=6)
    
    # Calculate edge widths based on weights
    max_weight = max(weights.values()) if weights else 1
    edge_widths = {edge: 1 + 3 * weights.get(edge, 0) / max_weight for edge in G.edges()}
    
    # Draw edges (passes)
    for edge in G.edges():
        from_player, to_player = edge
        if from_player in positions and to_player in positions:
            x1, y1 = positions[from_player]
            x2, y2 = positions[to_player]
            
            # Calculate edge width
            width = edge_widths.get(edge, 1)
            
            # Draw arrow
            plt.arrow(x1, y1, 0.8*(x2-x1), 0.8*(y2-y1), head_width=1.5, head_length=1.5, 
                     fc=node_color, ec=node_color, alpha=0.5, width=width/10, zorder=4)
    
    # Add title and labels
    plt.title(f'{team_name} Team Pass Network (Match {args.match_num})')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {team_name} team pass network visualization to {output_path}")

def analyze_team_possession(passes, ball_predictions):
    """
    Analyze team possession based on detected passes.
    
    Args:
        passes: List of detected passes
        ball_predictions: DataFrame with predicted ball coordinates
        
    Returns:
        possession_stats: Dictionary of possession statistics
    """
    # Count passes by team
    home_passes = [p for p in passes if p[2] and p[3]]  # Both from and to are home players
    away_passes = [p for p in passes if not p[2] and not p[3]]  # Both from and to are away players
    
    # Count frames where each team has possession
    home_possession_frames = set()
    away_possession_frames = set()
    
    for from_player, to_player, from_is_home, to_is_home, frame in passes:
        if from_is_home and to_is_home:
            # Add frames between consecutive home team passes
            if len(home_passes) > 1:
                for i in range(frame - 10, frame + 10):  # Approximate possession window
                    if 0 <= i < len(ball_predictions):
                        home_possession_frames.add(i)
        elif not from_is_home and not to_is_home:
            # Add frames between consecutive away team passes
            if len(away_passes) > 1:
                for i in range(frame - 10, frame + 10):  # Approximate possession window
                    if 0 <= i < len(ball_predictions):
                        away_possession_frames.add(i)
    
    # Calculate possession percentages
    total_frames = len(ball_predictions)
    home_possession_pct = len(home_possession_frames) / total_frames * 100
    away_possession_pct = len(away_possession_frames) / total_frames * 100
    
    # Calculate pass completion rates
    home_completion_rate = len(home_passes) / (len(home_passes) + len([p for p in passes if p[2] and not p[3]])) * 100 if home_passes else 0
    away_completion_rate = len(away_passes) / (len(away_passes) + len([p for p in passes if not p[2] and p[3]])) * 100 if away_passes else 0
    
    # Compile statistics
    possession_stats = {
        'home_passes': len(home_passes),
        'away_passes': len(away_passes),
        'home_possession_pct': home_possession_pct,
        'away_possession_pct': away_possession_pct,
        'home_completion_rate': home_completion_rate,
        'away_completion_rate': away_completion_rate
    }
    
    return possession_stats

def visualize_possession_stats(possession_stats, output_path):
    """
    Visualize team possession statistics.
    
    Args:
        possession_stats: Dictionary of possession statistics
        output_path: Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    plt.subplot(1, 2, 1)
    # Possession pie chart
    labels = ['Home Team', 'Away Team', 'Contested/Unknown']
    sizes = [
        possession_stats['home_possession_pct'],
        possession_stats['away_possession_pct'],
        100 - possession_stats['home_possession_pct'] - possession_stats['away_possession_pct']
    ]
    colors = ['blue', 'red', 'gray']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Ball Possession')
    
    plt.subplot(1, 2, 2)
    # Pass statistics bar chart
    teams = ['Home Team', 'Away Team']
    pass_counts = [possession_stats['home_passes'], possession_stats['away_passes']]
    completion_rates = [possession_stats['home_completion_rate'], possession_stats['away_completion_rate']]
    
    x = np.arange(len(teams))
    width = 0.35
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, pass_counts, width, label='Number of Passes', color=['blue', 'red'])
    bars2 = ax2.bar(x + width/2, completion_rates, width, label='Completion Rate (%)', color=['lightblue', 'lightcoral'])
    
    ax1.set_xlabel('Team')
    ax1.set_ylabel('Number of Passes')
    ax2.set_ylabel('Completion Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(teams)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('Pass Statistics')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved possession statistics visualization to {output_path}")

def create_ball_movement_heatmap(ball_predictions, home_df, away_df, output_path, field_length=105, field_width=68):
    """
    Create a heatmap of ball movement with player average positions.
    
    Args:
        ball_predictions: DataFrame with predicted ball coordinates
        home_df, away_df: DataFrames with player positions
        output_path: Path to save the visualization
        field_length, field_width: Field dimensions in meters
    """
    # Extract ball coordinates
    ball_x = ball_predictions['ball_x'].values / 10.0  # Scale to meters
    ball_y = ball_predictions['ball_y'].values / 10.0  # Scale to meters
    
    # Get average player positions
    home_positions = get_average_player_positions(home_df, away_df, 'home_')
    away_positions = get_average_player_positions(home_df, away_df, 'away_')
    
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
    
    # Create heatmap of ball positions
    heatmap, xedges, yedges = np.histogram2d(ball_x, ball_y, bins=50, range=[[0, field_length], [0, field_width]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Apply Gaussian filter for smoothing
    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap, sigma=1.0)
    
    # Create custom colormap (yellow to red)
    colors = [(1, 1, 0.3), (1, 0.5, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('yellow_to_red', colors, N=100)
    
    # Plot heatmap
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, alpha=0.7)
    plt.colorbar(label='Ball Density')
    
    # Draw home players
    for player_id, (x, y) in home_positions.items():
        plt.scatter(x, y, s=200, c='blue', alpha=0.7, edgecolors='black', zorder=5)
        label = player_id.replace('home_', '')
        plt.text(x, y + 1, label, fontsize=10, ha='center', va='center', zorder=6)
    
    # Draw away players
    for player_id, (x, y) in away_positions.items():
        plt.scatter(x, y, s=200, c='red', alpha=0.7, edgecolors='black', zorder=5)
        label = player_id.replace('away_', '')
        plt.text(x, y + 1, label, fontsize=10, ha='center', va='center', zorder=6)
    
    # Add title and labels
    plt.title(f'Ball Movement Heatmap with Player Positions (Match {args.match_num})')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ball movement heatmap visualization to {output_path}")

def main():
    """Main function."""
    global args
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load match data
    print(f"Loading match {args.match_num} data...")
    _, home_df, away_df = load_data(args.match_num, prediction_mode=True)
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_file}...")
    predictions = pd.read_csv(args.predictions_file)
    
    # Detect passes
    passes = detect_passes(predictions, home_df, away_df, args.possession_threshold)
    print(f"Detected {len(passes)} passes")
    
    # Create pass networks
    home_graph, home_weights = create_pass_network(passes, 'home_', args.min_passes)
    away_graph, away_weights = create_pass_network(passes, 'away_', args.min_passes)
    
    # Get average player positions
    home_positions = get_average_player_positions(home_df, away_df, 'home_')
    away_positions = get_average_player_positions(home_df, away_df, 'away_')
    
    # Visualize pass networks
    home_output_path = os.path.join(args.output_dir, f'Home_Team_Pass_Network_(Match_{args.match_num}).png')
    away_output_path = os.path.join(args.output_dir, f'Away_Team_Pass_Network_(Match_{args.match_num}).png')
    
    visualize_pass_network(home_graph, home_positions, home_weights, 'Home', home_output_path)
    visualize_pass_network(away_graph, away_positions, away_weights, 'Away', away_output_path)
    
    # Analyze team possession
    possession_stats = analyze_team_possession(passes, predictions)
    print("\nTeam Possession Analysis:")
    print(f"Home Team: {possession_stats['home_possession_pct']:.1f}% possession, {possession_stats['home_passes']} passes, {possession_stats['home_completion_rate']:.1f}% completion rate")
    print(f"Away Team: {possession_stats['away_possession_pct']:.1f}% possession, {possession_stats['away_passes']} passes, {possession_stats['away_completion_rate']:.1f}% completion rate")
    
    # Visualize possession statistics
    possession_output_path = os.path.join(args.output_dir, f'Team_Possession_Stats_(Match_{args.match_num}).png')
    visualize_possession_stats(possession_stats, possession_output_path)
    
    # Create ball movement heatmap
    heatmap_output_path = os.path.join(args.output_dir, f'Ball_Movement_Heatmap_(Match_{args.match_num}).png')
    create_ball_movement_heatmap(predictions, home_df, away_df, heatmap_output_path)

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
