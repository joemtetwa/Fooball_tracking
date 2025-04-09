
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calc_derivatives(data, dt=0.1):
    velocity = data.diff() / dt
    acceleration = velocity.diff() / dt
    return velocity.fillna(0), acceleration.fillna(0)

def compute_features(df_home, df_away, include_ball=True):
    df = df_home.merge(df_away, on=['MatchId', 'IdPeriod', 'Time'], suffixes=('_home', '_away'))

    ball_cols = ['ball_x', 'ball_y'] if include_ball else []
    player_cols_home = [col for col in df.columns if col.startswith('away_') and '_x' in col and col not in ball_cols]
    player_cols_away = [col for col in df.columns if col.startswith('home_') and '_x' in col and col not in ball_cols]
    player_ids_home = list(set([col.rsplit('_', 1)[0] for col in player_cols_home]))
    player_ids_away = list(set([col.rsplit('_', 1)[0] for col in player_cols_away]))

    tracking_cols = player_cols_home + player_cols_away
    tracking_y_cols = [col.replace('_x', '_y') for col in tracking_cols]
    all_cols = tracking_cols + tracking_y_cols + ball_cols
    df[all_cols] = df[all_cols].interpolate(limit_direction='both')

    for pid in player_ids_home + player_ids_away:
        if pid + '_x' in df.columns:
            vel_x, acc_x = calc_derivatives(df[pid + '_x'])
            vel_y, acc_y = calc_derivatives(df[pid + '_y'])
            df[pid + '_vx'] = vel_x
            df[pid + '_vy'] = vel_y
            df[pid + '_ax'] = acc_x
            df[pid + '_ay'] = acc_y

    if include_ball:
        for pid in player_ids_home + player_ids_away:
            dx = df[pid + '_x'] - df['ball_x']
            dy = df[pid + '_y'] - df['ball_y']
            df[pid + '_dist_to_ball'] = np.sqrt(dx**2 + dy**2)

        df['possessing_player'] = df[[pid + '_dist_to_ball' for pid in player_ids_home + player_ids_away]].idxmin(axis=1).str.replace('_dist_to_ball', '')
        df['possessing_team'] = df['possessing_player'].apply(lambda x: 'home' if x in player_ids_home else 'away')

    df['home_centroid_x'] = df[[pid + '_x' for pid in player_ids_home]].mean(axis=1)
    df['home_centroid_y'] = df[[pid + '_y' for pid in player_ids_home]].mean(axis=1)
    df['away_centroid_x'] = df[[pid + '_x' for pid in player_ids_away]].mean(axis=1)
    df['away_centroid_y'] = df[[pid + '_y' for pid in player_ids_away]].mean(axis=1)

    df['home_area'] = (df[[pid + '_x' for pid in player_ids_home]].max(axis=1) - df[[pid + '_x' for pid in player_ids_home]].min(axis=1)) *                       (df[[pid + '_y' for pid in player_ids_home]].max(axis=1) - df[[pid + '_y' for pid in player_ids_home]].min(axis=1))
    df['away_area'] = (df[[pid + '_x' for pid in player_ids_away]].max(axis=1) - df[[pid + '_x' for pid in player_ids_away]].min(axis=1)) *                       (df[[pid + '_y' for pid in player_ids_away]].max(axis=1) - df[[pid + '_y' for pid in player_ids_away]].min(axis=1))

    radius = 1000
    for pid in player_ids_home + player_ids_away:
        dx = df[pid + '_x'] - (df['ball_x'] if include_ball else 0)
        dy = df[pid + '_y'] - (df['ball_y'] if include_ball else 0)
        dist = np.sqrt(dx**2 + dy**2)
        df[pid + '_close'] = (dist < radius).astype(int)

    df['home_pressure'] = df[[pid + '_close' for pid in player_ids_home]].sum(axis=1)
    df['away_pressure'] = df[[pid + '_close' for pid in player_ids_away]].sum(axis=1)

    for lag in [1, 2, 3, 5, 10]:
        if include_ball:
            df[f'ball_x_lag_{lag}'] = df['ball_x'].shift(lag)
            df[f'ball_y_lag_{lag}'] = df['ball_y'].shift(lag)

    return df, player_ids_home + player_ids_away

def train_models(df, player_ids):
    feature_cols = [col for col in df.columns if any(sub in col for sub in ['_vx', '_vy', '_ax', '_ay', '_dist_to_ball', 'centroid', 'area', 'pressure', 'lag'])]
    df = df.dropna(subset=['ball_x', 'ball_y'] + feature_cols + ['possessing_player'])

    X = df[feature_cols]
    y_ball = df[['ball_x', 'ball_y']]
    y_possession = df['possessing_player']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_ball_train, y_ball_test, y_pos_train, y_pos_test = train_test_split(
        X_scaled, y_ball, y_possession, test_size=0.2, random_state=42)

    ball_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ball_model.fit(X_train, y_ball_train)

    pos_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pos_model.fit(X_train, y_pos_train)

    return ball_model, pos_model, scaler, feature_cols

def predict_match4(df_home, df_away, ball_model, pos_model, scaler, feature_cols):
    df, _ = compute_features(df_home, df_away, include_ball=False)
    df = df.fillna(0)

    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    ball_preds = ball_model.predict(X_scaled)
    pos_preds = pos_model.predict(X_scaled)

    result_df = df[['MatchId', 'IdPeriod', 'Time']].copy()
    result_df['ball_x_pred'] = ball_preds[:, 0]
    result_df['ball_y_pred'] = ball_preds[:, 1]
    result_df['possessing_player_pred'] = pos_preds
    result_df['possessing_team_pred'] = result_df['possessing_player_pred'].apply(lambda x: 'home' if '_home' in x else 'away')

    return result_df

# --- Example Usage ---

# Load match 1-3 for training
df_home_train = pd.read_excel("Home.xlsx", sheet_name="in")
df_away_train = pd.read_excel("Away.xlsx", sheet_name="in")
df_train, player_ids = compute_features(df_home_train, df_away_train)
ball_model, pos_model, scaler, feature_cols = train_models(df_train, player_ids)

# Load match 4 (without ball)
df_home_m4 = pd.read_excel("Match4_Home.xlsx")
df_away_m4 = pd.read_csv("Match4_Away.csv")
result_df = predict_match4(df_home_m4, df_away_m4, ball_model, pos_model, scaler, feature_cols)

# Save predictions
result_df.to_csv("match4_predictions.csv", index=False)
print("Predictions for Match 4 saved to match4_predictions.csv")
