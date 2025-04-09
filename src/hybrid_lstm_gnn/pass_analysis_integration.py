import numpy as np
import pandas as pd
import torch
import networkx as nx
from collections import defaultdict

def extract_pass_features(passes, player_id, team_prefix, window_size=10):
    """
    Extract pass-related features for a specific player.
    
    Args:
        passes: List of pass tuples (from_player, to_player, frame_start, frame_end)
        player_id: ID of the player to extract features for
        team_prefix: Prefix for the team ('home_' or 'away_')
        window_size: Number of recent passes to consider
        
    Returns:
        Dictionary of pass features
    """
    full_player_id = f"{team_prefix}{player_id}"
    
    # Count passes made and received
    passes_made = [p for p in passes if p[0] == full_player_id]
    passes_received = [p for p in passes if p[1] == full_player_id]
    
    # Get recent passes
    recent_passes_made = passes_made[-window_size:] if passes_made else []
    recent_passes_received = passes_received[-window_size:] if passes_received else []
    
    # Calculate pass frequencies
    pass_targets = defaultdict(int)
    for p in passes_made:
        pass_targets[p[1]] += 1
    
    # Most frequent pass targets
    frequent_targets = sorted(pass_targets.items(), key=lambda x: x[1], reverse=True)
    top_targets = frequent_targets[:3] if frequent_targets else []
    
    return {
        'total_passes_made': len(passes_made),
        'total_passes_received': len(passes_received),
        'recent_passes_made': len(recent_passes_made),
        'recent_passes_received': len(recent_passes_received),
        'top_targets': top_targets
    }

def calculate_player_influence(pass_matrix, centrality_metrics):
    """
    Calculate player influence scores based on pass matrix and centrality.
    
    Args:
        pass_matrix: DataFrame with pass probabilities
        centrality_metrics: Dictionary of player centrality values
        
    Returns:
        Dictionary mapping player IDs to influence scores
    """
    influence = {}
    
    for player in pass_matrix.index:
        # Combine passing frequency with centrality
        pass_frequency = pass_matrix.loc[player].sum()
        centrality = centrality_metrics.get(player, 0)
        
        # Calculate influence score (weighted combination)
        influence[player] = 0.7 * centrality + 0.3 * pass_frequency
    
    return influence

def create_pass_probability_matrix(passes, team_prefix):
    """
    Create a pass probability matrix for a team.
    
    Args:
        passes: List of pass tuples
        team_prefix: Team prefix ('home_' or 'away_')
        
    Returns:
        DataFrame with pass probabilities
    """
    # Handle empty passes list (for prediction mode)
    if not passes:
        print(f"No passes detected for {team_prefix} team. Creating empty pass probability matrix.")
        return pd.DataFrame()
    
    # Filter passes for the specified team
    team_passes = [p for p in passes if p[0].startswith(team_prefix) and p[1].startswith(team_prefix)]
    
    # Handle case where no passes for this team
    if not team_passes:
        print(f"No passes detected for {team_prefix} team. Creating empty pass probability matrix.")
        return pd.DataFrame()
    
    # Count passes between players
    pass_counts = defaultdict(int)
    player_pass_totals = defaultdict(int)
    
    for from_player, to_player, _, _ in team_passes:
        pass_counts[(from_player, to_player)] += 1
        player_pass_totals[from_player] += 1
    
    # Get unique players
    players = set()
    for from_player, to_player, _, _ in team_passes:
        players.add(from_player)
        players.add(to_player)
    
    # Create probability matrix
    players = sorted(list(players))
    matrix = pd.DataFrame(0, index=players, columns=players)
    
    for (from_player, to_player), count in pass_counts.items():
        if player_pass_totals[from_player] > 0:
            matrix.loc[from_player, to_player] = count / player_pass_totals[from_player]
    
    return matrix


def extract_pass_features_for_frame(passes, home_df, away_df, ball_df, idx, 
                                   home_pass_matrix, away_pass_matrix, window_size=10,
                                   prediction_mode=False):
    """
    Extract pass analysis features for a specific frame.
    
    Args:
        passes: List of pass tuples
        home_df, away_df: DataFrames with player positions
        ball_df: DataFrame with ball position
        idx: Current frame index
        home_pass_matrix, away_pass_matrix: Pass probability matrices
        window_size: Number of recent frames to consider
        prediction_mode: If True, handle the case where pass matrices are empty
        
    Returns:
        Tensor of pass features
    """
    
    # Handle empty pass matrices or prediction mode
    if prediction_mode or not passes or home_pass_matrix.empty or away_pass_matrix.empty:
        # Return default feature vector (all zeros) for prediction mode
        return np.zeros(10)  # Adjust size based on your feature extraction
    import numpy as np
    import torch
    from src.hybrid_lstm_gnn.data_utils import get_player_positions
    
    # Get ball position
    ball_pos = np.array([ball_df.iloc[idx]['ball_x'], ball_df.iloc[idx]['ball_y']])
    
    # Get player positions
    # We need to use our own implementation since the one in data_utils.py returns a different format
    # Get home player positions
    home_pos = []
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            pos_x = home_df.iloc[idx][x_col]
            pos_y = home_df.iloc[idx][y_col]
            if not np.isnan(pos_x) and not np.isnan(pos_y):
                home_pos.append((player_id, pos_x, pos_y))
    
    # Get away player positions
    away_pos = []
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            pos_x = away_df.iloc[idx][x_col]
            pos_y = away_df.iloc[idx][y_col]
            if not np.isnan(pos_x) and not np.isnan(pos_y):
                away_pos.append((player_id, pos_x, pos_y))
    
    # Find closest player to ball (potential possessor)
    min_dist = float('inf')
    possessor = None
    possessor_team = None
    
    for player in home_pos:
        player_x = player[1]
        player_y = player[2]
        dist = np.sqrt((ball_pos[0] - player_x)**2 + (ball_pos[1] - player_y)**2)
        if dist < min_dist:
            min_dist = dist
            possessor = player[0]
            possessor_team = 'home_'
    
    for player in away_pos:
        player_x = player[1]
        player_y = player[2]
        dist = np.sqrt((ball_pos[0] - player_x)**2 + (ball_pos[1] - player_y)**2)
        if dist < min_dist:
            min_dist = dist
            possessor = player[0]
            possessor_team = 'away_'
    
    # Get recent passes
    recent_passes = [p for p in passes if p[2] < idx and p[2] >= idx - window_size]
    
    # Extract features
    features = []
    
    if possessor:
        # Get pass matrix for possessor's team
        pass_matrix = home_pass_matrix if possessor_team == 'home_' else away_pass_matrix
        
        # 1. Possessor's passing tendencies
        if possessor in pass_matrix.index:
            # Top 3 pass targets and their probabilities
            pass_probs = pass_matrix.loc[possessor].sort_values(ascending=False)
            top_targets = pass_probs.head(3)
            
            for target, prob in top_targets.items():
                features.append(prob)
            
            # Pad if less than 3 targets
            features.extend([0] * (3 - len(top_targets)))
        else:
            features.extend([0, 0, 0])
        
        # 2. Recent pass frequency
        possessor_recent_passes = [p for p in recent_passes if p[0] == possessor]
        features.append(len(possessor_recent_passes) / max(1, len(recent_passes)))
        
        # 3. Pass reception frequency
        possessor_receptions = [p for p in recent_passes if p[1] == possessor]
        features.append(len(possessor_receptions) / max(1, len(recent_passes)))
    else:
        # No clear possessor
        features.extend([0, 0, 0, 0, 0])
    
    # 4. Team possession ratio
    home_possessions = [p for p in recent_passes if p[0].startswith('home_')]
    away_possessions = [p for p in recent_passes if p[0].startswith('away_')]
    
    home_ratio = len(home_possessions) / max(1, len(recent_passes))
    features.append(home_ratio)
    
    # 5. Pass sequence length
    current_sequence = []
    for p in reversed(passes):
        if p[2] <= idx:
            if not current_sequence or p[1] == current_sequence[0][0]:
                current_sequence.insert(0, p)
            else:
                break
    
    features.append(len(current_sequence) / 5.0)  # Normalize by typical sequence length
    
    # 6. Ball speed from passes
    if len(recent_passes) > 1:
        recent_pass = recent_passes[-1]
        pass_duration = recent_pass[3] - recent_pass[2]
        if pass_duration > 0:
            # Calculate ball speed during the pass
            start_pos = np.array([ball_df.iloc[recent_pass[2]]['ball_x'], 
                                 ball_df.iloc[recent_pass[2]]['ball_y']])
            end_pos = np.array([ball_df.iloc[recent_pass[3]]['ball_x'], 
                               ball_df.iloc[recent_pass[3]]['ball_y']])
            distance = np.linalg.norm(end_pos - start_pos)
            speed = distance / pass_duration
            features.append(min(1.0, speed / 20.0))  # Normalize by typical max speed
        else:
            features.append(0)
    else:
        features.append(0)
    
    # 7. Possession team (one-hot)
    features.append(1.0 if possessor_team == 'home_' else 0.0)
    features.append(1.0 if possessor_team == 'away_' else 0.0)
    
    return torch.tensor(features, dtype=torch.float32)


def create_enhanced_graph(home_df, away_df, ball_df, idx, home_pass_matrix, away_pass_matrix, 
                         proximity_threshold=5.0, pass_prob_threshold=0.1):
    """
    Create an enhanced graph incorporating both physical proximity and pass probabilities.
    
    Args:
        home_df, away_df: DataFrames with player positions
        ball_df: DataFrame with ball position
        idx: Current frame index
        home_pass_matrix, away_pass_matrix: Pass probability matrices
        proximity_threshold: Threshold for proximity-based edges
        pass_prob_threshold: Threshold for pass probability edges
        
    Returns:
        PyTorch Geometric Data object with enhanced graph
    """
    import torch
    from torch_geometric.data import Data
    from src.hybrid_lstm_gnn.data_utils import create_graph, get_player_positions
    
    # Create base graph with proximity edges
    base_graph = create_graph(home_df, away_df, ball_df, idx, proximity_threshold=proximity_threshold)
    
    if base_graph is None:
        return None
    
    # Extract node features and edge index
    node_features = base_graph.x
    edge_index = base_graph.edge_index.clone()
    edge_attr = torch.ones(edge_index.shape[1], 1)  # Default edge weight
    
    # Get player IDs at current frame
    # We need to use our own implementation since the one in data_utils.py returns a different format
    # Get home player positions
    home_pos = []
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            pos_x = home_df.iloc[idx][x_col]
            pos_y = home_df.iloc[idx][y_col]
            if not np.isnan(pos_x) and not np.isnan(pos_y):
                home_pos.append((player_id, pos_x, pos_y))
    
    # Get away player positions
    away_pos = []
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            pos_x = away_df.iloc[idx][x_col]
            pos_y = away_df.iloc[idx][y_col]
            if not np.isnan(pos_x) and not np.isnan(pos_y):
                away_pos.append((player_id, pos_x, pos_y))
    
    # Map node indices to player IDs
    node_to_player = {}
    for i, player in enumerate(home_pos):
        node_to_player[i] = player[0]
    
    for i, player in enumerate(away_pos):
        node_to_player[i + len(home_pos)] = player[0]
    
    # Add pass probability edges
    new_edges = []
    new_edge_weights = []
    
    # Process home team pass probabilities
    for i in range(len(home_pos)):
        from_player = node_to_player.get(i)
        if from_player not in home_pass_matrix.index:
            continue
            
        for j in range(len(home_pos)):
            to_player = node_to_player.get(j)
            if to_player not in home_pass_matrix.columns:
                continue
                
            prob = home_pass_matrix.loc[from_player, to_player]
            if prob > pass_prob_threshold:
                new_edges.append([i, j])
                new_edge_weights.append(prob)
    
    # Process away team pass probabilities
    offset = len(home_pos)
    for i in range(len(away_pos)):
        from_player = node_to_player.get(i + offset)
        if from_player not in away_pass_matrix.index:
            continue
            
        for j in range(len(away_pos)):
            to_player = node_to_player.get(j + offset)
            if to_player not in away_pass_matrix.columns:
                continue
                
            prob = away_pass_matrix.loc[from_player, to_player]
            if prob > pass_prob_threshold:
                new_edges.append([i + offset, j + offset])
                new_edge_weights.append(prob)
    
    # Add new edges to the graph
    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        new_weights_tensor = torch.tensor(new_edge_weights, dtype=torch.float).view(-1, 1)
        
        edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
        edge_attr = torch.cat([edge_attr, new_weights_tensor], dim=0)
    
    # Create enhanced graph
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)