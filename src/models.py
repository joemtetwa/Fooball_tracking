from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

class BallPossessionModel:
    def __init__(self, n_estimators=50, random_state=42, n_jobs=-1):
        # Create pipelines with imputation
        self.ball_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=1
            ))
        ])
        
        self.possession_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=1
            ))
        ])
        
        self.feature_cols = None
        self.training_columns = None
    
    def get_feature_columns(self, df, for_training=True):
        """Extract relevant feature columns from dataframe."""
        base_patterns = [
            '_vx', '_vy',      # Velocities
            'centroid',        # Team formation
            'area',           # Team spread
            '_speed'          # Speed features
        ]
        
        if for_training and 'ball_x' in df.columns:
            base_patterns.append('_dist_to_ball')
        
        features = []
        for pattern in base_patterns:
            features.extend([col for col in df.columns if pattern in col])
        
        return sorted(list(set(features)))  # Remove duplicates and sort for consistency
    
    def prepare_data(self, df, for_training=True):
        """Prepare features and target variables."""
        print("Selecting features...")
        if for_training:
            self.feature_cols = self.get_feature_columns(df, for_training=True)
            self.training_columns = sorted(list(df.columns))  # Store training columns
        else:
            # Use only features that were present during training
            available_features = set(self.feature_cols) & set(df.columns)
            missing_features = set(self.feature_cols) - set(df.columns)
            if missing_features:
                print(f"Warning: {len(missing_features)} features missing in prediction data")
                print("Missing features will be filled with zeros")
        
        print(f"Selected {len(self.feature_cols)} features")
        
        # Create feature matrix
        X = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0  # Fill missing features with zeros
        
        if for_training and 'ball_x' in df.columns:
            y_ball = df[['ball_x', 'ball_y']]
            y_possession = df['possessing_player']
            required_cols = ['ball_x', 'ball_y', 'possessing_player']
            df = df.dropna(subset=required_cols)  # Only drop NaN in target variables
        else:
            y_ball = None
            y_possession = None
        
        return X, y_ball, y_possession
    
    def fit(self, df):
        """Train both ball position and possession models."""
        print("Preparing training data...")
        X, y_ball, y_possession = self.prepare_data(df, for_training=True)
        
        # Split data
        print("Splitting train/test data...")
        X_train, X_test, y_ball_train, y_ball_test, y_pos_train, y_pos_test = train_test_split(
            X, y_ball, y_possession, test_size=0.2, random_state=42
        )
        
        # Train models
        print("\nTraining ball position model...")
        self.ball_pipeline.fit(X_train, y_ball_train)
        
        print("\nTraining possession model...")
        self.possession_pipeline.fit(X_train, y_pos_train)
        
        # Compute and print metrics
        print("\nComputing model performance...")
        ball_preds = self.ball_pipeline.predict(X_test)
        pos_preds = self.possession_pipeline.predict(X_test)
        
        ball_mae = np.mean(np.abs(ball_preds - y_ball_test.values))
        possession_acc = np.mean(pos_preds == y_pos_test)
        
        print(f"\nModel Performance:")
        print(f"Ball Position MAE: {ball_mae:.2f}")
        print(f"Possession Accuracy: {possession_acc:.2%}")
        
        return self
    
    def predict(self, df):
        """Generate predictions for new data."""
        if self.feature_cols is None:
            raise ValueError("Model has not been trained yet!")
        
        # Prepare features
        print("Preparing features for prediction...")
        X, _, _ = self.prepare_data(df, for_training=False)
        
        # Generate predictions
        print("Generating ball position predictions...")
        ball_preds = self.ball_pipeline.predict(X)
        
        print("Generating possession predictions...")
        pos_preds = self.possession_pipeline.predict(X)
        
        # Create results dataframe
        print("Preparing results...")
        result_df = df[['MatchId', 'IdPeriod', 'Time']].copy()
        result_df['ball_x_pred'] = ball_preds[:, 0]
        result_df['ball_y_pred'] = ball_preds[:, 1]
        result_df['possessing_player_pred'] = pos_preds
        result_df['possessing_team_pred'] = result_df['possessing_player_pred'].apply(
            lambda x: 'home' if 'home_' in x else 'away'
        )
        
        return result_df
    
    def get_feature_importance(self):
        """Get feature importance scores for both models."""
        if self.feature_cols is None:
            raise ValueError("Model has not been trained yet!")
        
        ball_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': np.mean(self.ball_pipeline.named_steps['model'].feature_importances_, axis=0)
        }).sort_values('importance', ascending=False)
        
        possession_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.possession_pipeline.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return ball_importance, possession_importance
