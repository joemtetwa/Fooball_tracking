import pandas as pd
import os

def analyze_excel_file(file_path):
    """Analyze an Excel file and print information about its structure."""
    print(f"\n{'='*50}")
    print(f"ANALYZING FILE: {os.path.basename(file_path)}")
    print(f"{'='*50}")
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Print basic information
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Print column names
    print("\nCOLUMN NAMES:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    if not missing_cols.empty:
        print("\nCOLUMNS WITH MISSING VALUES:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Print sample data (first 3 rows)
    print("\nSAMPLE DATA (first 3 rows):")
    print(df.head(3).to_string())
    
    # Check for ball coordinates
    ball_cols = [col for col in df.columns if 'ball' in col.lower()]
    if ball_cols:
        print("\nBALL COLUMNS:")
        for col in ball_cols:
            print(f"  {col}")
    
    # Check for player coordinates
    player_cols = [col for col in df.columns if ('home_' in col.lower() or 'away_' in col.lower()) and 
                  ('_x' in col.lower() or '_y' in col.lower())]
    
    if player_cols:
        print(f"\nPLAYER COORDINATE COLUMNS: {len(player_cols)}")
        # Get unique player IDs
        player_ids = set()
        for col in player_cols:
            parts = col.split('_')
            if len(parts) >= 2:
                player_ids.add(f"{parts[0]}_{parts[1]}")
        
        print(f"UNIQUE PLAYERS: {len(player_ids)}")
        print("PLAYER IDs:")
        for player_id in sorted(list(player_ids)):
            print(f"  {player_id}")
    
    return df

# Analyze both files
home_file = os.path.join("Data", "match_0", "Home.xlsx")
away_file = os.path.join("Data", "match_0", "Away.xlsx")

home_df = analyze_excel_file(home_file)
away_df = analyze_excel_file(away_file)

# Compare the files
print("\n" + "="*50)
print("COMPARISON BETWEEN FILES")
print("="*50)

# Check if the files have the same number of rows (timestamps)
print(f"Same number of rows: {len(home_df) == len(away_df)}")

# Check for common columns
common_cols = set(home_df.columns).intersection(set(away_df.columns))
print(f"\nCommon columns: {len(common_cols)}")
print("Common columns list:")
for col in sorted(list(common_cols)):
    print(f"  {col}")

# Check if ball coordinates are in both files
home_ball_cols = [col for col in home_df.columns if 'ball' in col.lower()]
away_ball_cols = [col for col in away_df.columns if 'ball' in col.lower()]

if home_ball_cols and away_ball_cols:
    print("\nBall coordinates present in both files")
    
    # Check if ball values are the same
    if set(home_ball_cols) == set(away_ball_cols):
        sample_ball_col = home_ball_cols[0]
        if home_df[sample_ball_col].equals(away_df[sample_ball_col]):
            print("Ball coordinates have identical values in both files")
        else:
            print("Ball coordinates have different values in the files")
            
            # Show a few examples
            print("\nSample ball coordinate comparison:")
            for i in range(min(5, len(home_df))):
                print(f"Row {i}: Home {sample_ball_col}={home_df[sample_ball_col].iloc[i]}, Away {sample_ball_col}={away_df[sample_ball_col].iloc[i]}")

# Analyze time column
if 'Time' in home_df.columns and 'Time' in away_df.columns:
    print("\nTime column analysis:")
    print(f"Time range in Home file: {home_df['Time'].min()} to {home_df['Time'].max()}")
    print(f"Time range in Away file: {away_df['Time'].min()} to {away_df['Time'].max()}")
    print(f"Time values are identical: {home_df['Time'].equals(away_df['Time'])}")
    
    # Check time intervals
    home_time_diffs = home_df['Time'].diff().dropna().unique()
    away_time_diffs = away_df['Time'].diff().dropna().unique()
    
    print(f"Time intervals in Home file: {sorted(home_time_diffs)}")
    print(f"Time intervals in Away file: {sorted(away_time_diffs)}")
