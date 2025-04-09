import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from src.hybrid_lstm_gnn.data_utils import get_player_positions

class BallPredictorLSTM(nn.Module):
    """LSTM model for predicting ball coordinates."""
    
    def __init__(self, input_size, hidden_size=128):
        """Initialize the LSTM model.
        
        Args:
            input_size: Number of features in the input
            hidden_size: Size of the hidden layers
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size//2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Output x, y coordinates
        )
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor of shape (batch_size, 2) with predicted ball coordinates
        """
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # We only need the output from the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out

def train_lstm_model(X_train, y_train, X_val, y_val, input_size, batch_size=32, epochs=50, learning_rate=0.001, model_save_path=None):
    """Train the LSTM model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        input_size: Number of features in the input
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the trained model
        
    Returns:
        Trained model and training history
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BallPredictorLSTM(input_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # If targets are sequences but we only predict the next position
            if len(targets.shape) > 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                
                # If targets are sequences but we only predict the next position
                if len(targets.shape) > 2 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs}, Time: {elapsed_time:.2f}s, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss and model_save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path} with validation loss: {val_loss:.6f}')
    
    # Load best model if saved
    if model_save_path and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    
    return model, history

def evaluate_lstm_model(model, X_test, y_test, batch_size=32):
    """Evaluate the LSTM model on test data.
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test targets
        batch_size: Batch size for evaluation
        
    Returns:
        Test loss and predictions
    """
    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create data loader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate
    criterion = nn.MSELoss()
    test_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            
            # If targets are sequences but we only predict the next position
            if len(targets.shape) > 2 and targets.shape[1] == 1:
                targets = targets.squeeze(1)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            # Store predictions
            predictions.append(outputs.numpy())
    
    test_loss /= len(test_loader.dataset)
    predictions = np.vstack(predictions)
    
    print(f'Test Loss: {test_loss:.6f}')
    
    return test_loss, predictions

def plot_training_history(history):
    """Plot training and validation loss history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_training_history.png')
    plt.close()

def predict_with_lstm(model, X, scaler=None):
    """Make predictions with the trained LSTM model.
    
    Args:
        model: Trained LSTM model
        X: Input features
        scaler: Scaler used to normalize the data
        
    Returns:
        Predicted ball coordinates
    """
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    return predictions.numpy()

def apply_player_proximity(predicted_pos, home_players, away_players, idx, proximity_threshold=15.0):
    """Adjust predicted ball positions based on player proximity.
    
    Args:
        predicted_pos: Predicted ball position
        home_players: DataFrame with home player positions
        away_players: DataFrame with away player positions
        idx: Current time index
        proximity_threshold: Distance threshold for player influence (meters)
        
    Returns:
        Adjusted ball position
    """
    # Get all player positions at current index
    all_players = []
    
    # Home players
    home_player_cols = [col for col in home_players.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_players.columns:
            x = home_players.iloc[idx][x_col]
            y = home_players.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                all_players.append((x, y, player_id))
    
    # Away players
    away_player_cols = [col for col in away_players.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_players.columns:
            x = away_players.iloc[idx][x_col]
            y = away_players.iloc[idx][y_col]
            if not np.isnan(x) and not np.isnan(y):
                all_players.append((x, y, player_id))
    
    if not all_players:
        return predicted_pos
    
    # Calculate distances to all players
    distances = []
    for player_pos in all_players:
        dist = np.sqrt((predicted_pos[0] - player_pos[0])**2 + (predicted_pos[1] - player_pos[1])**2)
        distances.append((dist, player_pos))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # If the closest player is within the proximity threshold, adjust the prediction
    if distances[0][0] < proximity_threshold:
        closest_player = distances[0][1]
        
        # Calculate influence weight based on distance (closer = more influence)
        influence = 1.0 - (distances[0][0] / proximity_threshold)
        
        # Adjust prediction (weighted average of prediction and player position)
        adjusted_x = predicted_pos[0] * (1 - influence) + closest_player[0] * influence
        adjusted_y = predicted_pos[1] * (1 - influence) + closest_player[1] * influence
        
        print(f"Ball prediction adjusted by player {closest_player[2]} at distance {distances[0][0]:.2f}m with influence {influence:.2f}")
        return np.array([adjusted_x, adjusted_y])
    
    return predicted_pos

def visualize_predictions(actual, predicted, home_players, away_players, start_idx, end_idx, history_length=5):
    """Visualize actual vs predicted ball positions with player positions.
    
    Args:
        actual: Actual ball positions
        predicted: Predicted ball positions
        home_players: DataFrame with home player positions
        away_players: DataFrame with away player positions
        start_idx: Start index for visualization
        end_idx: End index for visualization
        history_length: Number of previous positions to show
    """
    plt.figure(figsize=(12, 8))
    
    # Plot field boundaries (assuming standard dimensions)
    field_length = 105
    field_width = 68
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    
    # Draw halfway line
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Draw penalty areas
    plt.plot([0, 16.5, 16.5, 0], [field_width/2 - 20.15, field_width/2 - 20.15, field_width/2 + 20.15, field_width/2 + 20.15], 'k-')
    plt.plot([field_length, field_length - 16.5, field_length - 16.5, field_length], 
             [field_width/2 - 20.15, field_width/2 - 20.15, field_width/2 + 20.15, field_width/2 + 20.15], 'k-')
    
    # Draw goal areas
    plt.plot([0, 5.5, 5.5, 0], [field_width/2 - 9.15, field_width/2 - 9.15, field_width/2 + 9.15, field_width/2 + 9.15], 'k-')
    plt.plot([field_length, field_length - 5.5, field_length - 5.5, field_length], 
             [field_width/2 - 9.15, field_width/2 - 9.15, field_width/2 + 9.15, field_width/2 + 9.15], 'k-')
    
    # Draw penalty spots
    plt.plot(11, field_width/2, 'ko', markersize=4)
    plt.plot(field_length - 11, field_width/2, 'ko', markersize=4)
    
    # Plot actual ball trajectory
    actual_x = actual[start_idx:end_idx, 0]
    actual_y = actual[start_idx:end_idx, 1]
    plt.plot(actual_x, actual_y, 'b-', label='Actual Ball Path', linewidth=2)
    
    # Plot predicted ball trajectory
    pred_x = predicted[start_idx:end_idx, 0]
    pred_y = predicted[start_idx:end_idx, 1]
    plt.plot(pred_x, pred_y, 'r--', label='Predicted Ball Path', linewidth=2)
    
    # Plot players for the last frame
    # Home players
    home_player_cols = [col for col in home_players.columns if '_x' in col and 'home_' in col]
    
    for x_col in home_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in home_players.columns:
            x = home_players.iloc[end_idx-1][x_col]
            y = home_players.iloc[end_idx-1][y_col]
            if not np.isnan(x) and not np.isnan(y):
                plt.plot(x, y, 'bo', markersize=8)
    
    # Away players
    away_player_cols = [col for col in away_players.columns if '_x' in col and 'away_' in col]
    
    for x_col in away_player_cols:
        # Extract player ID from column name
        player_id = x_col.replace('_x', '')
        y_col = f"{player_id}_y"
        
        if y_col in away_players.columns:
            x = away_players.iloc[end_idx-1][x_col]
            y = away_players.iloc[end_idx-1][y_col]
            if not np.isnan(x) and not np.isnan(y):
                plt.plot(x, y, 'ro', markersize=8)
    
    # Plot the actual ball position for the last frame
    plt.plot(actual_x[-1], actual_y[-1], 'ko', markersize=6, label='Actual Ball')
    
    # Plot the predicted ball position for the last frame
    plt.plot(pred_x[-1], pred_y[-1], 'mo', markersize=6, label='Predicted Ball')
    
    # Plot ball history
    history_start = max(0, end_idx - history_length)
    for i in range(history_start, end_idx):
        alpha = 0.3 + 0.7 * (i - history_start) / (end_idx - history_start)
        plt.plot(actual[i, 0], actual[i, 1], 'ko', alpha=alpha, markersize=4)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Ball Position Prediction with Player Positions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('lstm_prediction_visualization.png')
    plt.close()

def create_demo_visualization(proximity_threshold=15.0):
    """Create a demonstration visualization to show player influence concept."""
    # Create artificial data for demonstration
    field_length = 105
    field_width = 68
    
    # Create a simple ball trajectory
    ball_x = np.linspace(20, 80, 50)
    ball_y = np.sin(ball_x / 10) * 10 + field_width / 2
    ball_trajectory = np.column_stack((ball_x, ball_y))
    
    # Create some player positions
    player_positions = [
        (30, field_width / 2 + 5, 'home_1'),
        (45, field_width / 2 - 8, 'away_2'),
        (60, field_width / 2 + 3, 'home_3'),
        (75, field_width / 2 - 6, 'away_4')
    ]
    
    # Create predicted trajectory without player influence
    predicted_no_influence = ball_trajectory.copy()
    predicted_no_influence[:, 1] += np.random.normal(0, 2, size=len(ball_trajectory))
    
    # Create predicted trajectory with player influence
    predicted_with_influence = predicted_no_influence.copy()
    
    # Apply player influence to each point
    influence_markers = []
    for i in range(len(predicted_with_influence)):
        # Find closest player
        min_dist = float('inf')
        closest_player = None
        
        for player in player_positions:
            dist = np.sqrt((predicted_with_influence[i, 0] - player[0])**2 + 
                           (predicted_with_influence[i, 1] - player[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_player = player
        
        # Apply influence if within threshold
        if min_dist < proximity_threshold:
            influence = 1.0 - (min_dist / proximity_threshold)
            original_pos = predicted_with_influence[i].copy()
            
            # Adjust prediction
            predicted_with_influence[i, 0] = predicted_with_influence[i, 0] * (1 - influence) + closest_player[0] * influence
            predicted_with_influence[i, 1] = predicted_with_influence[i, 1] * (1 - influence) + closest_player[1] * influence
            
            # Store information for visualization
            if i % 5 == 0:  # Only store some points to avoid clutter
                influence_markers.append((original_pos, predicted_with_influence[i], closest_player, influence))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Plot actual ball trajectory
    plt.plot(ball_trajectory[:, 0], ball_trajectory[:, 1], 'b-', label='Actual Ball Path', linewidth=2)
    
    # Plot predicted trajectory without player influence
    plt.plot(predicted_no_influence[:, 0], predicted_no_influence[:, 1], 'r--', 
             label='Predicted Path (No Player Influence)', linewidth=2)
    
    # Plot predicted trajectory with player influence
    plt.plot(predicted_with_influence[:, 0], predicted_with_influence[:, 1], 'g-', 
             label='Predicted Path (With Player Influence)', linewidth=2)
    
    # Plot players
    for player in player_positions:
        if 'home' in player[2]:
            plt.plot(player[0], player[1], 'bo', markersize=10, label='_nolegend_')
            plt.text(player[0] + 1, player[1] + 1, player[2], fontsize=9)
        else:
            plt.plot(player[0], player[1], 'ro', markersize=10, label='_nolegend_')
            plt.text(player[0] + 1, player[1] + 1, player[2], fontsize=9)
    
    # Plot influence markers
    for marker in influence_markers:
        original, adjusted, player, influence = marker
        
        # Draw line from original prediction to adjusted prediction
        plt.plot([original[0], adjusted[0]], [original[1], adjusted[1]], 'k:', alpha=0.5)
        
        # Draw line from player to adjusted prediction
        plt.plot([player[0], adjusted[0]], [player[1], adjusted[1]], 'm:', alpha=0.5)
        
        # Add influence weight text
        mid_x = (original[0] + adjusted[0]) / 2
        mid_y = (original[1] + adjusted[1]) / 2
        plt.text(mid_x, mid_y, f"{influence:.2f}", fontsize=8, ha='center', va='center', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Draw proximity circles around players
    for player in player_positions:
        proximity_circle = plt.Circle((player[0], player[1]), proximity_threshold, 
                                      fill=False, linestyle='--', alpha=0.3, 
                                      color='blue' if 'home' in player[2] else 'red')
        plt.gca().add_patch(proximity_circle)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Player Influence on Ball Trajectory Prediction')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('player_influence_demo.png')
    plt.close()
    
    print("Created demonstration visualization of player influence concept")
