import os
from data_processing import load_training_data, preprocess_data
from feature_engineering import engineer_all_features
from models import BallPossessionModel
from visualization import save_all_plots

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("Loading training data...")
    df_home, df_away = load_training_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df_home, df_away)
    
    print("Engineering features...")
    df, player_ids = engineer_all_features(df, include_ball=True)
    
    print("Training models...")
    model = BallPossessionModel(n_estimators=100)
    model.fit(df)
    
    # Save feature importance plots
    print("Generating feature importance plots...")
    save_all_plots(df, model)
    
    print("Training complete! Check the plots directory for visualization results.")

if __name__ == "__main__":
    main()
