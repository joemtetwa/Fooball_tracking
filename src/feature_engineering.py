import numpy as np
import pandas as pd

def calc_derivatives(data, dt=0.1):
    """Calculate velocity and acceleration from position data."""
    velocity = data.diff() / dt
    acceleration = velocity.diff() / dt
    return velocity.fillna(0), acceleration.fillna(0)

def get_player_ids(df):
    """Extract player IDs from dataframe columns."""
    player_cols = [col for col in df.columns if any(x in col for x in ['home_', 'away_']) and '_x' in col]
    player_ids = list(set([col.rsplit('_', 1)[0] for col in player_cols]))
    
    home_ids = [pid for pid in player_ids if pid.startswith('home_')]
    away_ids = [pid for pid in player_ids if pid.startswith('away_')]
    
    return home_ids, away_ids

def compute_player_features(df, player_ids):
    """Compute velocity and acceleration features for all players."""
    feature_data = {}
    
    for pid in player_ids:
        if pid + '_x' in df.columns and pid + '_y' in df.columns:
            # Calculate derivatives
            vel_x, acc_x = calc_derivatives(df[pid + '_x'])
            vel_y, acc_y = calc_derivatives(df[pid + '_y'])
            
            # Store features
            feature_data[f'{pid}_vx'] = vel_x
            feature_data[f'{pid}_vy'] = vel_y
            feature_data[f'{pid}_ax'] = acc_x
            feature_data[f'{pid}_ay'] = acc_y
            feature_data[f'{pid}_speed'] = np.sqrt(vel_x**2 + vel_y**2)
            feature_data[f'{pid}_acceleration'] = np.sqrt(acc_x**2 + acc_y**2)
    
    # Add all features at once to avoid fragmentation
    df = pd.concat([df, pd.DataFrame(feature_data)], axis=1)
    return df

def compute_team_features(df, player_ids_home, player_ids_away):
    """Compute team-level features like centroids and formation areas."""
    # Team centroids
    df['home_centroid_x'] = df[[pid + '_x' for pid in player_ids_home]].mean(axis=1)
    df['home_centroid_y'] = df[[pid + '_y' for pid in player_ids_home]].mean(axis=1)
    df['away_centroid_x'] = df[[pid + '_x' for pid in player_ids_away]].mean(axis=1)
    df['away_centroid_y'] = df[[pid + '_y' for pid in player_ids_away]].mean(axis=1)
    
    # Team spread/formation area
    df['home_area'] = (
        (df[[pid + '_x' for pid in player_ids_home]].max(axis=1) - 
         df[[pid + '_x' for pid in player_ids_home]].min(axis=1)) *
        (df[[pid + '_y' for pid in player_ids_home]].max(axis=1) - 
         df[[pid + '_y' for pid in player_ids_home]].min(axis=1))
    )
    
    df['away_area'] = (
        (df[[pid + '_x' for pid in player_ids_away]].max(axis=1) - 
         df[[pid + '_x' for pid in player_ids_away]].min(axis=1)) *
        (df[[pid + '_y' for pid in player_ids_away]].max(axis=1) - 
         df[[pid + '_y' for pid in player_ids_away]].min(axis=1))
    )
    
    return df

def compute_ball_features(df, player_ids, include_ball=True):
    """Compute ball-related features including distances and possession."""
    if not include_ball or 'ball_x' not in df.columns:
        return df
    
    feature_data = {}
    
    # Ball velocity and acceleration
    ball_vel_x, ball_acc_x = calc_derivatives(df['ball_x'])
    ball_vel_y, ball_acc_y = calc_derivatives(df['ball_y'])
    
    feature_data['ball_vx'] = ball_vel_x
    feature_data['ball_vy'] = ball_vel_y
    feature_data['ball_ax'] = ball_acc_x
    feature_data['ball_ay'] = ball_acc_y
    feature_data['ball_speed'] = np.sqrt(ball_vel_x**2 + ball_vel_y**2)
    
    # Distance to ball for each player
    for pid in player_ids:
        dx = df[pid + '_x'] - df['ball_x']
        dy = df[pid + '_y'] - df['ball_y']
        feature_data[f'{pid}_dist_to_ball'] = np.sqrt(dx**2 + dy**2)
    
    # Add all features at once
    df = pd.concat([df, pd.DataFrame(feature_data)], axis=1)
    
    # Determine possession based on closest player
    df['possessing_player'] = (
        df[[f'{pid}_dist_to_ball' for pid in player_ids]]
        .idxmin(axis=1)
        .str.replace('_dist_to_ball', '')
    )
    
    df['possessing_team'] = df['possessing_player'].apply(
        lambda x: 'home' if 'home_' in x else 'away'
    )
    
    return df

def create_temporal_features(df, include_ball=True):
    """Create time-based features including lagged variables."""
    feature_data = {}
    
    # Add time-based features
    for lag in [1, 2, 3, 5, 10]:
        if include_ball and 'ball_x' in df.columns:
            feature_data[f'ball_x_lag_{lag}'] = df['ball_x'].shift(lag)
            feature_data[f'ball_y_lag_{lag}'] = df['ball_y'].shift(lag)
            if 'ball_speed' in df.columns:
                feature_data[f'ball_speed_lag_{lag}'] = df['ball_speed'].shift(lag)
    
    # Add all features at once
    if feature_data:
        df = pd.concat([df, pd.DataFrame(feature_data)], axis=1)
    
    return df

def engineer_all_features(df, include_ball=True):
    """Main function to compute all features."""
    # Get player IDs
    player_ids_home, player_ids_away = get_player_ids(df)
    all_player_ids = player_ids_home + player_ids_away
    
    # Compute features
    df = compute_player_features(df, all_player_ids)
    df = compute_team_features(df, player_ids_home, player_ids_away)
    df = compute_ball_features(df, all_player_ids, include_ball)
    df = create_temporal_features(df, include_ball)
    
    return df, all_player_ids
