
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calc_derivatives(data, dt=0.1):
    velocity = data.diff() / dt
    acceleration = velocity.diff() / dt
    return velocity.fillna(0), acceleration.fillna(0)

def compute_features(df_home, df_away):
    df = df_home.merge(df_away, on=['MatchId', 'IdPeriod', 'Time'], suffixes=('_home', '_away'))
    ball_cols = ['ball_x', 'ball_y']
    df[ball_cols] = df[ball_cols].interpolate(limit_direction='both')

    player_cols_home = [col for col in df.columns if col.startswith('away_') and '_x' in col and col not in ball_cols]
    player_cols_away = [col for col in df.columns if col.startswith('home_') and '_x' in col and col not in ball_cols]
    player_ids_home = list(set([col.rsplit('_', 1)[0] for col in player_cols_home]))
    player_ids_away = list(set([col.rsplit('_', 1)[0] for col in player_cols_away]))

    tracking_cols = player_cols_home + player_cols_away
    tracking_y_cols = [col.replace('_x', '_y') for col in tracking_cols]
    df[tracking_cols + tracking_y_cols] = df[tracking_cols + tracking_y_cols].interpolate(limit_direction='both')

    for pid in player_ids_home + player_ids_away:
        if pid + '_x' in df.columns:
            vel_x, acc_x = calc_derivatives(df[pid + '_x'])
            vel_y, acc_y = calc_derivatives(df[pid + '_y'])
            df[pid + '_vx'] = vel_x
            df[pid + '_vy'] = vel_y
            df[pid + '_ax'] = acc_x
            df[pid + '_ay'] = acc_y

    for pid in player_ids_home + player_ids_away:
        dx = df[pid + '_x'] - df['ball_x']
        dy = df[pid + '_y'] - df['ball_y']
        df[pid + '_dist_to_ball'] = np.sqrt(dx**2 + dy**2)

    df['home_centroid_x'] = df[[pid + '_x' for pid in player_ids_home]].mean(axis=1)
    df['home_centroid_y'] = df[[pid + '_y' for pid in player_ids_home]].mean(axis=1)
    df['away_centroid_x'] = df[[pid + '_x' for pid in player_ids_away]].mean(axis=1)
    df['away_centroid_y'] = df[[pid + '_y' for pid in player_ids_away]].mean(axis=1)

    df['home_area'] = (df[[pid + '_x' for pid in player_ids_home]].max(axis=1) - df[[pid + '_x' for pid in player_ids_home]].min(axis=1)) *                       (df[[pid + '_y' for pid in player_ids_home]].max(axis=1) - df[[pid + '_y' for pid in player_ids_home]].min(axis=1))
    df['away_area'] = (df[[pid + '_x' for pid in player_ids_away]].max(axis=1) - df[[pid + '_x' for pid in player_ids_away]].min(axis=1)) *                       (df[[pid + '_y' for pid in player_ids_away]].max(axis=1) - df[[pid + '_y' for pid in player_ids_away]].min(axis=1))

    df['possessing_player'] = df[[pid + '_dist_to_ball' for pid in player_ids_home + player_ids_away]].idxmin(axis=1).str.replace('_dist_to_ball', '')
    df['possessing_team'] = df['possessing_player'].apply(lambda x: 'home' if x in player_ids_home else 'away')

    radius = 1000
    for pid in player_ids_home + player_ids_away:
        dx = df[pid + '_x'] - df['ball_x']
        dy = df[pid + '_y'] - df['ball_y']
        dist = np.sqrt(dx**2 + dy**2)
        df[pid + '_close'] = (dist < radius).astype(int)

    df['home_pressure'] = df[[pid + '_close' for pid in player_ids_home]].sum(axis=1)
    df['away_pressure'] = df[[pid + '_close' for pid in player_ids_away]].sum(axis=1)

    for lag in [1, 2, 3, 5, 10]:
        df[f'ball_x_lag_{lag}'] = df['ball_x'].shift(lag)
        df[f'ball_y_lag_{lag}'] = df['ball_y'].shift(lag)

    return df, player_ids_home + player_ids_away

def run_modeling_pipeline(df, player_ids):
    ball_cols = ['ball_x', 'ball_y']
    feature_cols = [col for col in df.columns if any(sub in col for sub in ['_vx', '_vy', '_ax', '_ay', '_dist_to_ball', 'centroid', 'area', 'pressure', 'lag'])]

    df = df.dropna(subset=ball_cols + feature_cols + ['possessing_player'])

    X = df[feature_cols]
    y_ball = df[ball_cols]
    y_possession = df['possessing_player']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_ball_train, y_ball_test, y_pos_train, y_pos_test = train_test_split(
        X_scaled, y_ball, y_possession, test_size=0.2, random_state=42)

    ball_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ball_model.fit(X_train, y_ball_train)
    y_ball_pred = ball_model.predict(X_test)

    pos_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pos_model.fit(X_train, y_pos_train)
    y_pos_pred = pos_model.predict(X_test)

    ball_mae = np.mean(np.abs(y_ball_pred - y_ball_test.values))
    possession_acc = np.mean(y_pos_pred == y_pos_test)

    print(f"Ball Position MAE: {ball_mae:.2f}")
    print(f"Possession Accuracy: {possession_acc:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_ball_test['ball_x'].values, y_ball_test['ball_y'].values, label='True Ball', alpha=0.7)
    plt.plot(y_ball_pred[:, 0], y_ball_pred[:, 1], label='Predicted Ball', linestyle='--', alpha=0.7)
    plt.title("Ball Trajectory: True vs Predicted")
    plt.xlabel("Ball X")
    plt.ylabel("Ball Y")
    plt.legend()
    plt.grid(True)
    plt.savefig("ball_trajectory_full_features.png")
    plt.show()

# Example usage (single match for now)
df_home = pd.read_excel("Home.xlsx", sheet_name="in")
df_away = pd.read_excel("Away.xlsx", sheet_name="in")
df_combined, all_players = compute_features(df_home, df_away)
run_modeling_pipeline(df_combined, all_players)
