# Soccer Ball Tracking System

This project implements an advanced ball coordinate prediction system for soccer matches using a hybrid LSTM-GNN model with enhanced features for pass analysis and player influence modeling.

## Project Overview

The system predicts ball coordinates in soccer matches by combining multiple prediction approaches:
- LSTM networks for temporal patterns in player and ball movement
- Graph Neural Networks (GNN) for spatial relationships between players
- Pass analysis for strategic passing patterns and probabilities
- Player influence modeling for realistic ball movement physics

## Project Structure

```
├── Data/                           # Match data directory
│   ├── match_0/                    # Training data with ball coordinates
│   ├── match_1/                    # Training data with ball coordinates
│   ├── match_2/                    # Training data with ball coordinates
│   ├── match_3/                    # Training data with ball coordinates
│   └── match_4/                    # Target match for prediction (no ball data)
├── src/                            # Source code
│   ├── hybrid_lstm_gnn/            # Hybrid model implementation
│   ├── lstm_model/                 # LSTM components
│   ├── data_processing.py          # Data loading and preprocessing
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── models.py                   # Model definitions
│   ├── player_tracking.py          # Player position analysis
│   ├── visualization.py            # Visualization utilities
│   └── simulation.py               # Ball physics simulation
├── models/                         # Saved trained models
├── predictions/                    # Output predictions
├── analysis/                       # Analysis results
├── pass_analysis/                  # Pass network analysis
├── enhanced_output/                # Enhanced prediction results
└── requirements.txt                # Project dependencies
```

## Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Structure**:
   - Each match directory contains `Home.xlsx` and `Away.xlsx` files with player tracking data
   - Matches 0-3 contain ball coordinate data for training
   - Match 4 is the target match for prediction (missing ball coordinates)

## Running the Ball Tracking Experiment

### 1. Data Preprocessing and Analysis

To analyze match data and understand player movements:

```bash
# Analyze a specific match (e.g., match 0)
python analyze_match_data.py

# Analyze player and ball proximity
python analyze_player_ball_proximity.py

# Create pass network visualizations
python create_pass_network_visualizations.py
```

### 2. Model Training

Train the enhanced hybrid LSTM-GNN model with pass analysis integration:

```bash
# Train the complete enhanced hybrid model
python test_enhanced_hybrid_model.py

# Alternative: Run the basic hybrid model
python run_hybrid_model.py
```

### 3. Ball Coordinate Prediction

Generate ball coordinate predictions for matches without ball data:

```bash
# Predict ball coordinates for match 4
python predict_match4.py

# Generate enhanced predictions with player influence
python enhance_ball_predictions.py
```

### 4. Prediction Analysis and Visualization

Analyze and visualize the prediction results:

```bash
# Analyze enhanced predictions
python analyze_enhanced_predictions.py

# Visualize predictions
python visualize_predictions.py

# Comprehensive movement analysis
python comprehensive_movement_analysis.py
```

## Model Components

### Enhanced Hybrid Model

The core of the system is an enhanced hybrid model that integrates:

1. **LSTM Component**: Processes temporal sequences of player movements
2. **GNN Component**: Models spatial relationships between players as a graph
3. **Pass Analysis Component**: Incorporates strategic passing patterns 
4. **Player Influence Model**: Adds realistic ball physics based on player positions

### Pass Analysis Integration

The pass analysis module:
- Creates pass probability matrices between players
- Extracts passing features for each frame
- Enhances graph structure with pass probability edges

### Player Influence Modeling

The player influence component:
- Calculates influence vectors based on player positions
- Implements possession detection
- Applies realistic ball physics simulation

## Analysis Tools

The system includes several analysis tools:

1. **Enhanced Prediction Analysis**:
   - Compares original and enhanced predictions
   - Calculates metrics like speed, acceleration, and direction changes
   - Generates heatmaps and trajectory visualizations

2. **Pass Network Analysis**:
   - Visualizes passing networks between players
   - Calculates pass probabilities
   - Identifies common passing sequences

3. **Player Movement Analysis**:
   - Analyzes player positions and formations
   - Calculates player influence zones
   - Evaluates team strategies

## Performance Metrics

The enhanced model maintains high correlation with baseline predictions while providing more realistic ball movement:
- Coordinate correlation: X: 0.9989, Y: 0.9975
- Improved smoothness in ball trajectory
- Realistic response to player positions and team dynamics

## Configuration Options

Most scripts accept command-line arguments to customize their behavior:

```bash
# Example with custom parameters
python analyze_enhanced_predictions.py --original_file predictions/match_4_ball_predictions.csv --enhanced_file predictions/match_4_enhanced_predictions.csv --output_dir analysis_enhanced
```

Check each script's `parse_args()` function for available options.
