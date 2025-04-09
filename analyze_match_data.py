import pandas as pd
import os
import numpy as np

def analyze_excel_file(file_path):
    """Analyze an Excel file and print information about its structure."""
    print(f"\nAnalyzing file: {file_path}")
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Print basic information
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Print column names and data types
    print("\nColumn Names and Data Types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"  {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Print sample data (first 5 rows)
    print("\nSample Data (first 5 rows):")
    print(df.head())
    
    # Identify coordinate columns
    coord_cols = [col for col in df.columns if '_x' in col or '_y' in col]
    print(f"\nCoordinate Columns: {len(coord_cols)}")
    for col in coord_cols[:10]:  # Print first 10 coordinate columns
        print(f"  {col}")
    
    if len(coord_cols) > 10:
        print(f"  ... and {len(coord_cols) - 10} more")
    
    # Check if ball coordinates are in the file
    ball_cols = [col for col in df.columns if 'ball' in col.lower()]
    print(f"\nBall Columns: {len(ball_cols)}")
    for col in ball_cols:
        print(f"  {col}")
    
    # Analyze coordinate ranges
    if coord_cols:
        print("\nCoordinate Ranges:")
        for col in coord_cols[:5]:  # Print ranges for first 5 coordinate columns
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  {col}: Min={min_val:.2f}, Max={max_val:.2f}")
    
    return df

# Analyze both files
home_file = os.path.join("Data", "match_0", "Home.xlsx")
away_file = os.path.join("Data", "match_0", "Away.xlsx")

home_df = analyze_excel_file(home_file)
away_df = analyze_excel_file(away_file)

# Check if the files have the same number of rows (timestamps)
print("\nComparison:")
print(f"Home file rows: {len(home_df)}")
print(f"Away file rows: {len(away_df)}")
print(f"Same number of rows: {len(home_df) == len(away_df)}")

# Check for ball coordinates in both files
home_ball_cols = [col for col in home_df.columns if 'ball' in col.lower()]
away_ball_cols = [col for col in away_df.columns if 'ball' in col.lower()]

print("\nBall columns in Home file:", home_ball_cols)
print("Ball columns in Away file:", away_ball_cols)

# Check for player coordinate patterns
print("\nPlayer Coordinate Patterns:")
home_player_pattern = [col for col in home_df.columns if 'home_' in col.lower() and ('_x' in col.lower() or '_y' in col.lower())]
away_player_pattern = [col for col in away_df.columns if 'away_' in col.lower() and ('_x' in col.lower() or '_y' in col.lower())]
home_in_away = [col for col in away_df.columns if 'home_' in col.lower()]
away_in_home = [col for col in home_df.columns if 'away_' in col.lower()]

print(f"Home player coordinates in Home file: {len(home_player_pattern)}")
print(f"Away player coordinates in Away file: {len(away_player_pattern)}")
print(f"Home player coordinates in Away file: {len(home_in_away)}")
print(f"Away player coordinates in Home file: {len(away_in_home)}")

# Determine if ball coordinates are in both files or just one
if home_ball_cols and away_ball_cols:
    print("\nBall coordinates present in both files")
elif home_ball_cols:
    print("\nBall coordinates only in Home file")
elif away_ball_cols:
    print("\nBall coordinates only in Away file")
else:
    print("\nNo ball coordinates found in either file")

# Check for timestamp column
print("\nTimestamp Information:")
time_cols = [col for col in home_df.columns if 'time' in col.lower() or 'frame' in col.lower()]
print(f"Potential time columns: {time_cols}")
if time_cols:
    for col in time_cols:
        if col in home_df.columns:
            print(f"  {col} in Home file: {home_df[col].dtype}")
            print(f"  First few values: {home_df[col].head()}")
