import pandas as pd
import numpy as np
import os

def load_match_data(match_number):
    """Load home and away data for a specific match."""
    base_path = os.path.join('Data', f'match_{match_number}')
    
    # Try different file extensions
    extensions = [('.xlsx', pd.read_excel), ('.csv', pd.read_csv)]
    df_home = df_away = None
    
    for ext, reader in extensions:
        try:
            home_path = os.path.join(base_path, f'Home{ext}')
            away_path = os.path.join(base_path, f'Away{ext}')
            
            if os.path.exists(home_path) and os.path.exists(away_path):
                df_home = reader(home_path)
                df_away = reader(away_path)
                break
        except Exception as e:
            print(f"Error loading {ext} files for match {match_number}: {str(e)}")
            continue
    
    if df_home is None or df_away is None:
        print(f"Could not load data for match {match_number}")
        return None, None
    
    # Add match identifier
    df_home['MatchId'] = match_number
    df_away['MatchId'] = match_number
    
    return df_home, df_away

def load_training_data():
    """Load and combine data from matches 0-3 for training."""
    all_home_data = []
    all_away_data = []
    
    for match_num in range(4):  # Matches 0-3
        df_home, df_away = load_match_data(match_num)
        if df_home is not None and df_away is not None:
            all_home_data.append(df_home)
            all_away_data.append(df_away)
    
    if not all_home_data or not all_away_data:
        raise ValueError("No training data could be loaded!")
    
    combined_home = pd.concat(all_home_data, ignore_index=True)
    combined_away = pd.concat(all_away_data, ignore_index=True)
    
    return combined_home, combined_away

def preprocess_data(df_home, df_away):
    """Basic preprocessing of home and away dataframes."""
    if df_home is None or df_away is None:
        raise ValueError("Cannot preprocess None dataframes")
        
    # Merge home and away data
    df = df_home.merge(df_away, on=['MatchId', 'IdPeriod', 'Time'], 
                      suffixes=('', '_away'))
    
    # Handle missing values in tracking data
    tracking_cols = [col for col in df.columns if '_x' in col or '_y' in col]
    df[tracking_cols] = df[tracking_cols].interpolate(limit_direction='both')
    
    return df
