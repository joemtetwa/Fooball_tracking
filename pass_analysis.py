import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import networkx as nx
from collections import defaultdict, Counter

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

def get_player_positions(home_df, away_df, idx):
    """Extract all player positions at a given index."""
    # Get home player positions
    home_pos = {}
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            x = home_df.iloc[idx][x_col]
            y = home_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                home_pos[player_id] = (x, y)
    
    # Get away player positions
    away_pos = {}
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            x = away_df.iloc[idx][x_col]
            y = away_df.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                away_pos[player_id] = (x, y)
    
    return home_pos, away_pos

def detect_ball_possession(ball_pos, home_pos, away_pos, possession_radius=2.0):
    """Detect which player has possession of the ball based on proximity."""
    min_dist = float('inf')
    possessing_player = None
    
    # Check home players
    for player_id, pos in home_pos.items():
        dist = euclidean(ball_pos, pos)
        if dist < min_dist:
            min_dist = dist
            possessing_player = player_id
    
    # Check away players
    for player_id, pos in away_pos.items():
        dist = euclidean(ball_pos, pos)
        if dist < min_dist:
            min_dist = dist
            possessing_player = player_id
    
    # Only return a player if they're within the possession radius
    if min_dist <= possession_radius:
        return possessing_player
    return None

def detect_passes_prediction_mode(home_df, away_df):
    """
    Create empty pass data for prediction mode when ball coordinates are not available.
    
    Parameters:
    -----------
    home_df, away_df : DataFrame
        DataFrames containing player coordinates
    
    Returns:
    --------
    passes : list
        Empty list of passes for compatibility with prediction mode
    """
    return []

def detect_passes(ball_df, home_df, away_df, possession_radius=2.0, ball_speed_threshold=5.0, 
                 min_pass_distance=5.0, max_frames_between_possessions=15, prediction_mode=False):
    """
    Detect passes between players based on ball movement and player positions.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame containing ball coordinates
    home_df, away_df : DataFrame
        DataFrames containing player coordinates
    possession_radius : float
        Maximum distance between player and ball to be considered in possession
    ball_speed_threshold : float
        Minimum ball speed to be considered a potential pass
    min_pass_distance : float
        Minimum distance between players to be considered a pass
    max_frames_between_possessions : int
        Maximum number of frames between consecutive possessions to be considered a pass
    prediction_mode : bool
        If True, handle the case where ball coordinates are not available
    
    Returns:
    --------
    passes : list
        List of tuples (from_player, to_player, frame_start, frame_end)
    """
    
    # In prediction mode, if ball_df has no real coordinates, return empty passes
    if prediction_mode and ('ball_x' not in ball_df.columns or 'ball_y' not in ball_df.columns or 
                           ball_df['ball_x'].sum() == 0 and ball_df['ball_y'].sum() == 0):
        print("No ball coordinates available in prediction mode. Returning empty passes list.")
        return []
    passes = []
    ball_speeds = []
    possessions = []
    
    # Calculate ball speeds
    for i in range(1, len(ball_df)):
        prev_x, prev_y = ball_df.iloc[i-1]['ball_x'], ball_df.iloc[i-1]['ball_y']
        curr_x, curr_y = ball_df.iloc[i]['ball_x'], ball_df.iloc[i]['ball_y']
        
        # Calculate displacement and speed
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Time difference (assuming constant frame rate)
        dt = 1.0
        
        # Speed in units per frame
        speed = displacement / dt
        ball_speeds.append(speed)
    
    # Add a placeholder for the first frame
    ball_speeds.insert(0, 0)
    
    # Detect possessions for each frame
    for i in range(len(ball_df)):
        ball_pos = (ball_df.iloc[i]['ball_x'], ball_df.iloc[i]['ball_y'])
        home_pos, away_pos = get_player_positions(home_df, away_df, i)
        
        possessing_player = detect_ball_possession(ball_pos, home_pos, away_pos, possession_radius)
        possessions.append((i, possessing_player, ball_speeds[i]))
    
    # Detect passes by analyzing possession changes
    last_possession = None
    possession_start = None
    
    for i, (frame, player, ball_speed) in enumerate(possessions):
        # If we have a player with possession
        if player is not None:
            # If this is a new possession
            if last_possession is None or last_possession != player:
                # If we had a previous possession, this might be a pass
                if last_possession is not None and possession_start is not None:
                    # Check if the time between possessions is reasonable
                    frames_between = frame - possession_start
                    if frames_between <= max_frames_between_possessions:
                        # Check if the ball moved fast enough during the transition
                        max_ball_speed = max(ball_speeds[possession_start:frame+1])
                        if max_ball_speed >= ball_speed_threshold:
                            # Get positions of both players at the time of possession change
                            home_pos_start, away_pos_start = get_player_positions(home_df, away_df, possession_start)
                            home_pos_end, away_pos_end = get_player_positions(home_df, away_df, frame)
                            
                            # Combine dictionaries for easier lookup
                            all_pos_start = {**home_pos_start, **away_pos_start}
                            all_pos_end = {**home_pos_end, **away_pos_end}
                            
                            # Check if both players are in the position dictionaries
                            if last_possession in all_pos_start and player in all_pos_end:
                                # Calculate distance between players
                                player_distance = euclidean(all_pos_start[last_possession], all_pos_end[player])
                                
                                # If distance is significant, consider it a pass
                                if player_distance >= min_pass_distance:
                                    passes.append((last_possession, player, possession_start, frame))
            
            # Update possession tracking
            last_possession = player
            possession_start = frame
        # If no player has possession but ball is moving fast, continue tracking
        elif ball_speed < ball_speed_threshold:
            # Reset possession tracking if ball is not moving
            last_possession = None
            possession_start = None
    
    return passes

def calculate_pass_probabilities(passes, team_prefix='home_'):
    """
    Calculate pass probabilities between players of the specified team.
    
    Parameters:
    -----------
    passes : list
        List of tuples (from_player, to_player, frame_start, frame_end)
    team_prefix : str
        Prefix to filter players of a specific team
    
    Returns:
    --------
    pass_matrix : DataFrame
        Matrix of pass probabilities between players
    """
    # Filter passes for the specified team
    team_passes = [p for p in passes if p[0].startswith(team_prefix) and p[1].startswith(team_prefix)]
    
    # Count passes between players
    pass_counts = defaultdict(Counter)
    for from_player, to_player, _, _ in team_passes:
        pass_counts[from_player][to_player] += 1
    
    # Get unique player IDs
    players = sorted(set([p for p, _ in pass_counts.items()] + 
                       [p for _ in pass_counts.values() for p in _.keys()]))
    
    # Create pass matrix
    pass_matrix = pd.DataFrame(0, index=players, columns=players)
    
    # Fill in pass counts
    for from_player, to_counts in pass_counts.items():
        for to_player, count in to_counts.items():
            pass_matrix.loc[from_player, to_player] = count
    
    # Calculate probabilities (normalize by row)
    for player in players:
        total_passes = pass_matrix.loc[player].sum()
        if total_passes > 0:
            pass_matrix.loc[player] = pass_matrix.loc[player] / total_passes
    
    return pass_matrix

def visualize_pass_network(pass_matrix, player_positions, title="Pass Network", min_probability=0.05):
    """
    Visualize the pass network between players.
    
    Parameters:
    -----------
    pass_matrix : DataFrame
        Matrix of pass probabilities between players
    player_positions : dict
        Dictionary mapping player IDs to their average positions (x, y)
    title : str
        Title for the plot
    min_probability : float
        Minimum probability to display an edge
    """
    plt.figure(figsize=(12, 8))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with positions
    for player, pos in player_positions.items():
        if player in pass_matrix.index:
            # Extract player number for display
            player_num = player.split('_')[1]
            G.add_node(player, pos=pos, label=player_num)
    
    # Add edges with weights
    for from_player in pass_matrix.index:
        for to_player in pass_matrix.columns:
            prob = pass_matrix.loc[from_player, to_player]
            if prob >= min_probability and from_player != to_player:
                G.add_edge(from_player, to_player, weight=prob, width=prob*5)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw field
    field_length = 105
    field_width = 68
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.8)
    
    # Draw node labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Draw edges with varying width based on probability
    edges = G.edges(data=True)
    weights = [data['weight'] for _, _, data in edges]
    widths = [data['width'] for _, _, data in edges]
    
    # Create a colormap for edge colors based on weight
    edge_colors = [plt.cm.Blues(w) for w in weights]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color=edge_colors, 
                          arrowsize=15, connectionstyle='arc3,rad=0.1', alpha=0.7)
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(False)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pass network visualization saved to {title.replace(' ', '_')}.png")
    
    return G

def calculate_average_player_positions(home_df, away_df, start_frame=0, end_frame=None):
    """
    Calculate average positions for all players over a range of frames.
    
    Parameters:
    -----------
    home_df, away_df : DataFrame
        DataFrames containing player coordinates
    start_frame, end_frame : int
        Range of frames to consider
    
    Returns:
    --------
    avg_positions : dict
        Dictionary mapping player IDs to their average positions (x, y)
    """
    if end_frame is None:
        end_frame = len(home_df)
    
    # Initialize position sums and counts
    position_sums = defaultdict(lambda: np.array([0.0, 0.0]))
    position_counts = defaultdict(int)
    
    # Sum positions over frames
    for idx in range(start_frame, end_frame):
        home_pos, away_pos = get_player_positions(home_df, away_df, idx)
        
        # Add home player positions
        for player_id, pos in home_pos.items():
            position_sums[player_id] += np.array(pos)
            position_counts[player_id] += 1
        
        # Add away player positions
        for player_id, pos in away_pos.items():
            position_sums[player_id] += np.array(pos)
            position_counts[player_id] += 1
    
    # Calculate averages
    avg_positions = {}
    for player_id, pos_sum in position_sums.items():
        if position_counts[player_id] > 0:
            avg_positions[player_id] = tuple(pos_sum / position_counts[player_id])
    
    return avg_positions

def analyze_ball_possession_sequences(passes, min_sequence_length=3):
    """
    Analyze sequences of consecutive passes to identify common patterns.
    
    Parameters:
    -----------
    passes : list
        List of tuples (from_player, to_player, frame_start, frame_end)
    min_sequence_length : int
        Minimum length of pass sequences to consider
    
    Returns:
    --------
    sequences : dict
        Dictionary mapping sequence patterns to their counts
    """
    # Sort passes by start frame
    sorted_passes = sorted(passes, key=lambda x: x[2])
    
    # Build pass chains
    chains = []
    current_chain = []
    
    for i, (from_player, to_player, start_frame, end_frame) in enumerate(sorted_passes):
        if not current_chain:
            # Start a new chain
            current_chain = [from_player, to_player]
        else:
            # Check if this pass continues the chain
            if to_player == current_chain[-1] and i > 0:
                # Check if frames are close enough
                prev_end_frame = sorted_passes[i-1][3]
                time_gap = start_frame - prev_end_frame
                
                if time_gap <= 30:  # Arbitrary threshold for continuity
                    current_chain.append(to_player)
                else:
                    # Save the current chain if it's long enough
                    if len(current_chain) >= min_sequence_length:
                        chains.append(tuple(current_chain))
                    # Start a new chain
                    current_chain = [from_player, to_player]
            else:
                # Save the current chain if it's long enough
                if len(current_chain) >= min_sequence_length:
                    chains.append(tuple(current_chain))
                # Start a new chain
                current_chain = [from_player, to_player]
    
    # Add the last chain if it's long enough
    if len(current_chain) >= min_sequence_length:
        chains.append(tuple(current_chain))
    
    # Count occurrences of each sequence
    sequence_counts = Counter(chains)
    
    return sequence_counts

def analyze_passes(match_num=0, start_frame=1000, num_frames=10000, 
                  possession_radius=2.0, ball_speed_threshold=5.0):
    """
    Analyze passes for a specific match.
    
    Parameters:
    -----------
    match_num : int
        Match number to analyze
    start_frame : int
        Starting frame for analysis
    num_frames : int
        Number of frames to analyze
    possession_radius : float
        Maximum distance between player and ball to be considered in possession
    ball_speed_threshold : float
        Minimum ball speed to be considered a potential pass
    """
    # Create output directory
    os.makedirs("pass_analysis", exist_ok=True)
    
    # Load data
    print(f"Analyzing passes for match {match_num}...")
    ball_df, home_df, away_df = load_data(match_num)
    
    # Limit to specified frames
    end_frame = min(start_frame + num_frames, len(ball_df))
    ball_df = ball_df.iloc[start_frame:end_frame].reset_index(drop=True)
    home_df = home_df.iloc[start_frame:end_frame].reset_index(drop=True)
    away_df = away_df.iloc[start_frame:end_frame].reset_index(drop=True)
    
    print(f"Analyzing frames {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
    
    # Detect passes
    print("Detecting passes...")
    passes = detect_passes(ball_df, home_df, away_df, 
                          possession_radius=possession_radius,
                          ball_speed_threshold=ball_speed_threshold)
    
    print(f"Detected {len(passes)} passes")
    
    # Calculate average player positions
    print("Calculating average player positions...")
    avg_positions = calculate_average_player_positions(home_df, away_df)
    
    # Calculate pass probabilities for home team
    print("Calculating pass probabilities for home team...")
    home_pass_matrix = calculate_pass_probabilities(passes, team_prefix='home_')
    
    # Calculate pass probabilities for away team
    print("Calculating pass probabilities for away team...")
    away_pass_matrix = calculate_pass_probabilities(passes, team_prefix='away_')
    
    # Visualize pass networks
    print("Visualizing pass networks...")
    home_avg_positions = {p: pos for p, pos in avg_positions.items() if p.startswith('home_')}
    away_avg_positions = {p: pos for p, pos in avg_positions.items() if p.startswith('away_')}
    
    home_G = visualize_pass_network(home_pass_matrix, home_avg_positions, 
                                   title=f"Home Team Pass Network (Match {match_num})")
    away_G = visualize_pass_network(away_pass_matrix, away_avg_positions, 
                                   title=f"Away Team Pass Network (Match {match_num})")
    
    # Analyze pass sequences
    print("Analyzing pass sequences...")
    home_passes = [p for p in passes if p[0].startswith('home_') and p[1].startswith('home_')]
    away_passes = [p for p in passes if p[0].startswith('away_') and p[1].startswith('away_')]
    
    home_sequences = analyze_ball_possession_sequences(home_passes)
    away_sequences = analyze_ball_possession_sequences(away_passes)
    
    # Print top pass sequences
    print("\nTop 5 Home Team Pass Sequences:")
    for seq, count in home_sequences.most_common(5):
        print(f"  Sequence: {' -> '.join([p.split('_')[1] for p in seq])}, Count: {count}")
    
    print("\nTop 5 Away Team Pass Sequences:")
    for seq, count in away_sequences.most_common(5):
        print(f"  Sequence: {' -> '.join([p.split('_')[1] for p in seq])}, Count: {count}")
    
    # Create heatmaps of pass probabilities
    plt.figure(figsize=(10, 8))
    sns.heatmap(home_pass_matrix, annot=True, cmap='Blues', fmt='.2f', 
               xticklabels=[p.split('_')[1] for p in home_pass_matrix.columns],
               yticklabels=[p.split('_')[1] for p in home_pass_matrix.index])
    plt.title(f"Home Team Pass Probability Matrix (Match {match_num})")
    plt.tight_layout()
    plt.savefig(f"pass_analysis/home_pass_matrix_match_{match_num}.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(away_pass_matrix, annot=True, cmap='Reds', fmt='.2f',
               xticklabels=[p.split('_')[1] for p in away_pass_matrix.columns],
               yticklabels=[p.split('_')[1] for p in away_pass_matrix.index])
    plt.title(f"Away Team Pass Probability Matrix (Match {match_num})")
    plt.tight_layout()
    plt.savefig(f"pass_analysis/away_pass_matrix_match_{match_num}.png", dpi=300)
    plt.close()
    
    # Calculate player-specific metrics
    print("\nCalculating player-specific metrics...")
    
    # Calculate total passes made and received
    home_passes_made = home_pass_matrix.sum(axis=1)
    home_passes_received = home_pass_matrix.sum(axis=0)
    
    away_passes_made = away_pass_matrix.sum(axis=1)
    away_passes_received = away_pass_matrix.sum(axis=0)
    
    # Calculate centrality metrics for pass networks
    home_centrality = nx.eigenvector_centrality_numpy(home_G, weight='weight')
    away_centrality = nx.eigenvector_centrality_numpy(away_G, weight='weight')
    
    # Create player metrics dataframes
    home_metrics = pd.DataFrame({
        'Player': [p.split('_')[1] for p in home_pass_matrix.index],
        'Passes Made': home_passes_made.values,
        'Passes Received': home_passes_received.values,
        'Centrality': [home_centrality.get(p, 0) for p in home_pass_matrix.index]
    })
    
    away_metrics = pd.DataFrame({
        'Player': [p.split('_')[1] for p in away_pass_matrix.index],
        'Passes Made': away_passes_made.values,
        'Passes Received': away_passes_received.values,
        'Centrality': [away_centrality.get(p, 0) for p in away_pass_matrix.index]
    })
    
    # Sort by centrality
    home_metrics = home_metrics.sort_values('Centrality', ascending=False)
    away_metrics = away_metrics.sort_values('Centrality', ascending=False)
    
    # Print player metrics
    print("\nHome Team Player Metrics (Top 5 by Centrality):")
    print(home_metrics.head(5).to_string(index=False))
    
    print("\nAway Team Player Metrics (Top 5 by Centrality):")
    print(away_metrics.head(5).to_string(index=False))
    
    # Save metrics to CSV
    home_metrics.to_csv(f"pass_analysis/home_player_metrics_match_{match_num}.csv", index=False)
    away_metrics.to_csv(f"pass_analysis/away_player_metrics_match_{match_num}.csv", index=False)
    
    print("\nAnalysis complete! Results saved to pass_analysis/ directory.")
    
    return passes, home_pass_matrix, away_pass_matrix, home_G, away_G

if __name__ == "__main__":
    # Run pass analysis for match 0
    analyze_passes(match_num=0, start_frame=1000, num_frames=10000)
