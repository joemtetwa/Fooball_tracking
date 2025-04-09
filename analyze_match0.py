import pandas as pd
import os

# Define paths
home_file = os.path.join("Data", "match_0", "Home.xlsx")
away_file = os.path.join("Data", "match_0", "Away.xlsx")

# Read the files
print("Reading Home.xlsx...")
home_df = pd.read_excel(home_file)
print("Reading Away.xlsx...")
away_df = pd.read_excel(away_file)

# Basic information
print("\n--- Basic Information ---")
print(f"Home file shape: {home_df.shape}")
print(f"Away file shape: {away_df.shape}")

# Check for ball columns
home_ball_cols = [col for col in home_df.columns if 'ball' in col.lower()]
away_ball_cols = [col for col in away_df.columns if 'ball' in col.lower()]

print("\n--- Ball Columns ---")
print(f"Ball columns in Home file: {home_ball_cols}")
print(f"Ball columns in Away file: {away_ball_cols}")

# Check first few rows of ball data
if home_ball_cols:
    print("\n--- Sample Ball Data (Home) ---")
    print(home_df[home_ball_cols].head())

# Identify player columns
home_player_cols = [col for col in home_df.columns if 'home_' in col.lower() and ('_x' in col.lower() or '_y' in col.lower())]
away_player_cols = [col for col in away_df.columns if 'away_' in col.lower() and ('_x' in col.lower() or '_y' in col.lower())]

# Get unique player IDs
home_player_ids = set()
for col in home_player_cols:
    parts = col.split('_')
    if len(parts) >= 2:
        home_player_ids.add(f"{parts[0]}_{parts[1]}")

away_player_ids = set()
for col in away_player_cols:
    parts = col.split('_')
    if len(parts) >= 2:
        away_player_ids.add(f"{parts[0]}_{parts[1]}")

print("\n--- Player Information ---")
print(f"Home player coordinate columns: {len(home_player_cols)}")
print(f"Away player coordinate columns: {len(away_player_cols)}")
print(f"Unique home players: {len(home_player_ids)}")
print(f"Unique away players: {len(away_player_ids)}")

# Print some player IDs
print("\n--- Sample Player IDs ---")
print("Home players:", sorted(list(home_player_ids))[:5], "...")
print("Away players:", sorted(list(away_player_ids))[:5], "...")

# Check if both files contain the same time values
print("\n--- Time Information ---")
if 'Time' in home_df.columns and 'Time' in away_df.columns:
    print(f"Time range in Home file: {home_df['Time'].min()} to {home_df['Time'].max()}")
    print(f"Time range in Away file: {away_df['Time'].min()} to {away_df['Time'].max()}")
    print(f"Time values are identical: {home_df['Time'].equals(away_df['Time'])}")
    
    # Check time intervals
    home_time_diffs = home_df['Time'].diff().dropna().value_counts().sort_index()
    print(f"Time intervals in Home file: {dict(home_time_diffs)}")

# Check for missing values
print("\n--- Missing Values ---")
home_missing = home_df.isnull().sum()
away_missing = away_df.isnull().sum()

print(f"Columns with missing values in Home file: {sum(home_missing > 0)}")
print(f"Columns with missing values in Away file: {sum(away_missing > 0)}")

# Sample data structure
print("\n--- Sample Data Structure ---")
print("Home file columns (first 10):", list(home_df.columns)[:10])
print("Away file columns (first 10):", list(away_df.columns)[:10])

# Check if the files have the same structure
common_cols = set(home_df.columns).intersection(set(away_df.columns))
print(f"\nCommon columns between files: {len(common_cols)}")
print(f"Columns only in Home file: {len(set(home_df.columns) - common_cols)}")
print(f"Columns only in Away file: {len(set(away_df.columns) - common_cols)}")

# Print a cleaner sample of the data
print("\n--- Clean Sample Data (First 3 rows) ---")
sample_cols = ['MatchId', 'Time']
if home_ball_cols:
    sample_cols.extend(home_ball_cols)
if home_player_cols:
    sample_cols.extend(home_player_cols[:4])  # First 2 players (x,y)

print(home_df[sample_cols].head(3))
