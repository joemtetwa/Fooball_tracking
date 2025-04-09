import os
import pandas as pd
import traceback
from data_processing import load_match_data, preprocess_data, load_training_data
from feature_engineering import engineer_all_features
from models import BallPossessionModel
from visualization import save_all_plots

def main():
    try:
        # Load and preprocess match 4 data
        print("Loading match 4 data...")
        df_home, df_away = load_match_data(4)
        if df_home is None or df_away is None:
            raise ValueError("Failed to load match 4 data")
        print(f"Loaded match 4 data - Home shape: {df_home.shape}, Away shape: {df_away.shape}")
        
        print("\nPreprocessing match 4 data...")
        df = preprocess_data(df_home, df_away)
        print(f"Preprocessed data shape: {df.shape}")
        
        print("\nEngineering features for match 4...")
        df, player_ids = engineer_all_features(df, include_ball=False)
        print(f"Feature engineered data shape: {df.shape}")
        print(f"Number of players: {len(player_ids)}")
        
        # Load training data and train model
        print("\nTraining model on matches 0-3...")
        df_home_train, df_away_train = load_training_data()
        print(f"Loaded training data - Home shape: {df_home_train.shape}, Away shape: {df_away_train.shape}")
        
        df_train = preprocess_data(df_home_train, df_away_train)
        print(f"Preprocessed training data shape: {df_train.shape}")
        
        df_train, _ = engineer_all_features(df_train, include_ball=True)
        print(f"Feature engineered training data shape: {df_train.shape}")
        
        model = BallPossessionModel()
        model.fit(df_train)
        
        # Generate predictions for match 4
        print("\nGenerating predictions for match 4...")
        predictions = model.predict(df)
        print(f"Generated predictions shape: {predictions.shape}")
        
        # Save predictions
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'match4_predictions.csv')
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        
        # Generate visualization plots
        print("\nGenerating visualization plots...")
        save_all_plots(predictions, model)
        
        print("\nPrediction pipeline completed successfully!")
        print("- Predictions saved to predictions/match4_predictions.csv")
        print("- Visualization plots saved to plots directory")
        
    except Exception as e:
        print("\nError occurred during prediction pipeline:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
