import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(match_num, prediction_mode=False):
    """Load match data from Excel or CSV files.
    
    Args:
        match_num: Match number to load
        prediction_mode: If True, allows loading data without ball coordinates for prediction
    
    Returns:
        ball_df, home_df, away_df: DataFrames with match data
    """
    data_dir = f"Data/match_{match_num}"
    home_file_xlsx = f"{data_dir}/Home.xlsx"
    away_file_xlsx = f"{data_dir}/Away.xlsx"
    home_file_csv = f"{data_dir}/Home.csv"
    away_file_csv = f"{data_dir}/Away.csv"
    
    # Try loading Excel files first, then CSV if Excel not available
    if os.path.exists(home_file_xlsx) and os.path.exists(away_file_xlsx):
        print(f"Reading {home_file_xlsx}")
        print(f"Reading {away_file_xlsx}")
        home_df = pd.read_excel(home_file_xlsx)
        away_df = pd.read_excel(away_file_xlsx)
    elif os.path.exists(home_file_csv) and os.path.exists(away_file_csv):
        print(f"Reading {home_file_csv}")
        print(f"Reading {away_file_csv}")
        home_df = pd.read_csv(home_file_csv)
        away_df = pd.read_csv(away_file_csv)
    else:
        raise FileNotFoundError(f"Could not find match data files for match {match_num}")
    
    # Extract ball coordinates if available
    ball_df = pd.DataFrame()
    if 'ball_x' in home_df.columns and 'ball_y' in home_df.columns:
        ball_df['ball_x'] = home_df['ball_x']
        ball_df['ball_y'] = home_df['ball_y']
        ball_df['ball_z'] = home_df['ball_z'] if 'ball_z' in home_df.columns else 0
    elif not prediction_mode:
        # Only raise error if not in prediction mode
        raise ValueError(f"Ball coordinates not found in match {match_num} data")
    else:
        # In prediction mode, create empty ball dataframe with same length as player data
        print(f"Ball coordinates not found in match {match_num} data. Creating empty ball dataframe for prediction.")
        ball_df = pd.DataFrame({
            'ball_x': [0] * len(home_df),
            'ball_y': [0] * len(home_df),
            'ball_z': [0] * len(home_df)
        })
    
    # Handle missing values
    home_df = home_df.fillna(0)
    away_df = away_df.fillna(0)
    ball_df = ball_df.fillna(0)
    
    print(f"Ball coordinates shape: {ball_df.shape}")
    print(f"Home players shape: {home_df.shape}")
    print(f"Away players shape: {away_df.shape}")
    
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

def preprocess_data_lstm(home_df, away_df, ball_df, n_steps=5, n_future=1, scaler=None):
    """Preprocess data for LSTM model."""
    # Extract player coordinates
    home_coords, away_coords = extract_player_coordinates(home_df, away_df)
    
    # Calculate features: player positions, velocities, distances to ball, etc.
    features = []
    
    for t in range(len(ball_df)):
        # Skip if we don't have enough previous frames for velocity
        if t < 1:
            continue
            
        # Get ball position
        ball_pos = ball_df.iloc[t][['ball_x', 'ball_y']].values
        
        # Skip if ball position contains NaN
        if np.isnan(ball_pos).any():
            continue
        
        # Reshape player coordinates
        home_players = []
        away_players = []
        
        # Process home players
        for player_id in set([col.split('_x')[0] for col in home_coords.columns if '_x' in col]):
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in home_coords.columns and y_col in home_coords.columns:
                x = home_coords.iloc[t][x_col]
                y = home_coords.iloc[t][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    home_players.append([x, y])
        
        # Process away players
        for player_id in set([col.split('_x')[0] for col in away_coords.columns if '_x' in col]):
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in away_coords.columns and y_col in away_coords.columns:
                x = away_coords.iloc[t][x_col]
                y = away_coords.iloc[t][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    away_players.append([x, y])
        
        # Convert to numpy arrays
        home_players = np.array(home_players)
        away_players = np.array(away_players)
        
        # Skip if we don't have enough players
        if len(home_players) < 1 or len(away_players) < 1:
            continue
        
        # Velocity (from previous frame)
        prev_ball_pos = ball_df.iloc[t-1][['ball_x', 'ball_y']].values
        ball_vel = ball_pos - prev_ball_pos
        
        # Calculate previous player positions for velocity
        prev_home_players = []
        prev_away_players = []
        
        # Process home players for previous frame
        for player_id in set([col.split('_x')[0] for col in home_coords.columns if '_x' in col]):
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in home_coords.columns and y_col in home_coords.columns:
                x = home_coords.iloc[t-1][x_col]
                y = home_coords.iloc[t-1][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    prev_home_players.append([x, y])
        
        # Process away players for previous frame
        for player_id in set([col.split('_x')[0] for col in away_coords.columns if '_x' in col]):
            x_col = f"{player_id}_x"
            y_col = f"{player_id}_y"
            
            if x_col in away_coords.columns and y_col in away_coords.columns:
                x = away_coords.iloc[t-1][x_col]
                y = away_coords.iloc[t-1][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    prev_away_players.append([x, y])
        
        # Convert to numpy arrays
        prev_home_players = np.array(prev_home_players)
        prev_away_players = np.array(prev_away_players)
        
        # Skip if player counts don't match (can happen with substitutions)
        if len(prev_home_players) != len(home_players) or len(prev_away_players) != len(away_players):
            continue
        
        # Calculate player velocities
        home_vel = home_players - prev_home_players
        away_vel = away_players - prev_away_players
        
        # Distance from each player to the ball
        home_dist_to_ball = [np.linalg.norm(p - ball_pos) for p in home_players]
        away_dist_to_ball = [np.linalg.norm(p - ball_pos) for p in away_players]
        
        # Determine possession (player closest to ball)
        min_home_dist = min(home_dist_to_ball) if home_dist_to_ball else float('inf')
        min_away_dist = min(away_dist_to_ball) if away_dist_to_ball else float('inf')
        possession = 0 if min_home_dist < min_away_dist else 1  # 0=home team, 1=away team
        
        # Combine features
        feature = np.concatenate([
            ball_pos,                # Ball position
            ball_vel,                # Ball velocity
            home_players.flatten(),  # Home player positions
            away_players.flatten(),  # Away player positions
            home_vel.flatten(),      # Home player velocities
            away_vel.flatten(),      # Away player velocities
            [min(min_home_dist, min_away_dist)],  # Minimum distance to ball
            [possession]             # Team possession
        ])
        
        features.append(feature)
    
    # Skip if we don't have enough features
    if len(features) < n_steps + n_future:
        raise ValueError(f"Not enough valid data points after preprocessing. Only {len(features)} features available.")
    
    # Scale features
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        # Use provided scaler
        scaled_features = scaler.transform(features)
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(n_steps, len(scaled_features) - n_future):
        X.append(scaled_features[i-n_steps:i])
        
        # Target is the future ball position
        future_idx = i + n_future
        if future_idx < len(ball_df):
            y.append(ball_df.iloc[future_idx][['ball_x', 'ball_y']].values)
    
    return np.array(X), np.array(y), scaler

def create_graph(home_df, away_df, ball_df, idx, prev_idx=None, proximity_threshold=5.0):
    """Create a graph for a single frame with players and ball as nodes."""
    # Get ball position
    ball_pos = ball_df.iloc[idx][['ball_x', 'ball_y']].values
    
    # Get previous ball position if available
    prev_ball_pos = ball_df.iloc[prev_idx][['ball_x', 'ball_y']].values if prev_idx is not None else None
    
    # Node features: [x, y, velocity_x, velocity_y, team (0=home, 1=away, 2=ball), distance to ball]
    nodes = []
    
    # Process home players
    home_player_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_df.columns:
            x = home_df.iloc[idx][x_col]
            y = home_df.iloc[idx][y_col]
            
            if not np.isnan(x) and not np.isnan(y):
                pos = np.array([x, y])
                
                # Calculate velocity if previous frame is available
                if prev_idx is not None:
                    prev_x = home_df.iloc[prev_idx][x_col]
                    prev_y = home_df.iloc[prev_idx][y_col]
                    
                    if not np.isnan(prev_x) and not np.isnan(prev_y):
                        prev_pos = np.array([prev_x, prev_y])
                        vel = pos - prev_pos
                    else:
                        vel = np.zeros(2)
                else:
                    vel = np.zeros(2)
                
                # Calculate distance to ball
                dist_to_ball = np.linalg.norm(pos - ball_pos)
                
                # Add node: [x, y, vx, vy, team, dist_to_ball]
                nodes.append([pos[0], pos[1], vel[0], vel[1], 0, dist_to_ball])
    
    # Process away players
    away_player_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_df.columns:
            x = away_df.iloc[idx][x_col]
            y = away_df.iloc[idx][y_col]
            
            if not np.isnan(x) and not np.isnan(y):
                pos = np.array([x, y])
                
                # Calculate velocity if previous frame is available
                if prev_idx is not None:
                    prev_x = away_df.iloc[prev_idx][x_col]
                    prev_y = away_df.iloc[prev_idx][y_col]
                    
                    if not np.isnan(prev_x) and not np.isnan(prev_y):
                        prev_pos = np.array([prev_x, prev_y])
                        vel = pos - prev_pos
                    else:
                        vel = np.zeros(2)
                else:
                    vel = np.zeros(2)
                
                # Calculate distance to ball
                dist_to_ball = np.linalg.norm(pos - ball_pos)
                
                # Add node: [x, y, vx, vy, team, dist_to_ball]
                nodes.append([pos[0], pos[1], vel[0], vel[1], 1, dist_to_ball])
    
    # Add ball node
    ball_vel = ball_pos - prev_ball_pos if prev_ball_pos is not None else np.zeros(2)
    nodes.append([ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1], 2, 0])
    
    # Skip if we don't have enough nodes
    if len(nodes) < 3:  # At least one player from each team and the ball
        return None
    
    # Create edge connections based on proximity
    edge_index = []
    node_positions = np.array([n[:2] for n in nodes])
    num_nodes = len(nodes)
    
    # Connect players within proximity threshold of each other
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if np.linalg.norm(node_positions[i] - node_positions[j]) < proximity_threshold:
                    edge_index.append([i, j])
    
    # Always connect the ball to the closest player
    ball_idx = num_nodes - 1
    player_distances = [np.linalg.norm(node_positions[i] - node_positions[ball_idx]) for i in range(num_nodes - 1)]
    closest_player = np.argmin(player_distances)
    
    # Ensure connection between ball and closest player
    edge_index.append([ball_idx, closest_player])
    edge_index.append([closest_player, ball_idx])
    
    # Convert to PyTorch tensors
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def prepare_gnn_dataset(home_df, away_df, ball_df, sequence_length=5, prediction_steps=1):
    """Prepare dataset for GNN training with sequences of graphs."""
    # Create list to store graph sequences and targets
    graph_sequences = []
    targets = []
    
    for t in range(sequence_length, len(ball_df) - prediction_steps):
        # Create sequence of graphs
        sequence = []
        valid_sequence = True
        
        for i in range(t - sequence_length, t):
            # Create graph for this frame
            prev_i = i - 1 if i > 0 else None
            graph = create_graph(home_df, away_df, ball_df, i, prev_i)
            
            # Skip if graph creation failed
            if graph is None:
                valid_sequence = False
                break
                
            sequence.append(graph)
        
        # Skip if any graph in the sequence is invalid
        if not valid_sequence or len(sequence) != sequence_length:
            continue
            
        # Add sequence to dataset
        graph_sequences.append(sequence)
        
        # Target is the future ball position
        future_idx = t + prediction_steps
        if future_idx < len(ball_df):
            target = ball_df.iloc[future_idx][['ball_x', 'ball_y']].values
            targets.append(target)
    
    return graph_sequences, np.array(targets)

def save_processed_data(X, y, scaler, output_dir, prefix="lstm"):
    """Save processed data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{prefix}_X.npy"), X)
    np.save(os.path.join(output_dir, f"{prefix}_y.npy"), y)
    
    # Save scaler if provided
    if scaler is not None:
        import joblib
        joblib.dump(scaler, os.path.join(output_dir, f"{prefix}_scaler.joblib"))
    
    print(f"Saved processed data to {output_dir}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

def load_processed_data(input_dir, prefix="lstm"):
    """Load processed data from disk."""
    X = np.load(os.path.join(input_dir, f"{prefix}_X.npy"))
    y = np.load(os.path.join(input_dir, f"{prefix}_y.npy"))
    
    # Load scaler if available
    scaler_path = os.path.join(input_dir, f"{prefix}_scaler.joblib")
    if os.path.exists(scaler_path):
        import joblib
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    print(f"Loaded processed data from {input_dir}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, scaler
