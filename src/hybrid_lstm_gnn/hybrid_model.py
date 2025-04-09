import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data import DataLoader, TensorDataset

from src.hybrid_lstm_gnn.lstm_model import BallPredictorLSTM
from src.hybrid_lstm_gnn.gnn_model import TemporalGNN

class HybridLSTMGNN(nn.Module):
    """Hybrid model combining LSTM and GNN for ball coordinate prediction."""
    
    def __init__(self, lstm_input_size, gnn_input_dim, lstm_hidden_size=128, gnn_hidden_dim=64, 
                 fusion_hidden_size=64, use_attention=True):
        """Initialize the hybrid model.
        
        Args:
            lstm_input_size: Number of features for LSTM input
            gnn_input_dim: Number of node features for GNN
            lstm_hidden_size: Size of LSTM hidden layers
            gnn_hidden_dim: Size of GNN hidden layers
            fusion_hidden_size: Size of fusion layer
            use_attention: Whether to use attention in GNN
        """
        super().__init__()
        
        # LSTM component
        self.lstm = BallPredictorLSTM(lstm_input_size, lstm_hidden_size)
        
        # GNN component
        self.gnn = TemporalGNN(gnn_input_dim, gnn_hidden_dim, lstm_hidden_size, 
                              gnn_hidden_dim, use_attention)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(gnn_hidden_dim + 2, fusion_hidden_size),  # GNN output + LSTM output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_size, 2)  # Final output: x, y coordinates
        )
    
    def forward(self, lstm_input, graph_sequence):
        """Forward pass through the network.
        
        Args:
            lstm_input: Input tensor for LSTM of shape (batch_size, sequence_length, input_size)
            graph_sequence: List of PyTorch Geometric Data objects for GNN
            
        Returns:
            Tensor with predicted ball coordinates
        """
        # Get LSTM prediction
        lstm_output = self.lstm(lstm_input)
        
        # Get GNN prediction (features, not coordinates)
        gnn_features = self.gnn.forward(graph_sequence)
        
        # Concatenate LSTM and GNN outputs
        combined = torch.cat((lstm_output, gnn_features), dim=1)
        
        # Final prediction through fusion layer
        prediction = self.fusion(combined)
        
        return prediction

def train_hybrid_model(lstm_data, gnn_sequences, targets, batch_size=32, epochs=50, 
                       learning_rate=0.001, model_save_path=None):
    """Train the hybrid LSTM-GNN model.
    
    Args:
        lstm_data: LSTM input data
        gnn_sequences: List of graph sequences for GNN
        targets: Target ball coordinates
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the trained model
        
    Returns:
        Trained model and training history
    """
    # Determine input dimensions
    lstm_input_size = lstm_data.shape[2]
    gnn_input_dim = gnn_sequences[0][0].x.shape[1]
    
    # Initialize model
    model = HybridLSTMGNN(lstm_input_size, gnn_input_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert LSTM data and targets to PyTorch tensors
    lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Split data into training and validation sets (80/20 split)
    n_samples = len(lstm_tensor)
    n_train = int(0.8 * n_samples)
    
    train_lstm = lstm_tensor[:n_train]
    train_gnn = gnn_sequences[:n_train]
    train_targets = targets_tensor[:n_train]
    
    val_lstm = lstm_tensor[n_train:]
    val_gnn = gnn_sequences[n_train:]
    val_targets = targets_tensor[n_train:]
    
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
        
        # Process in batches
        for i in range(0, len(train_lstm), batch_size):
            batch_lstm = train_lstm[i:i + batch_size]
            batch_gnn = train_gnn[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass (process each sample individually for GNN)
            batch_predictions = []
            for j in range(len(batch_lstm)):
                prediction = model(batch_lstm[j:j+1], batch_gnn[j])
                batch_predictions.append(prediction)
            
            # Stack predictions
            batch_predictions = torch.cat(batch_predictions, dim=0)
            
            # Compute loss
            loss = criterion(batch_predictions, batch_targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_lstm)
        
        train_loss /= len(train_lstm)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_lstm), batch_size):
                batch_lstm = val_lstm[i:i + batch_size]
                batch_gnn = val_gnn[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]
                
                # Forward pass (process each sample individually for GNN)
                batch_predictions = []
                for j in range(len(batch_lstm)):
                    prediction = model(batch_lstm[j:j+1], batch_gnn[j])
                    batch_predictions.append(prediction)
                
                # Stack predictions
                batch_predictions = torch.cat(batch_predictions, dim=0)
                
                # Compute loss
                loss = criterion(batch_predictions, batch_targets)
                val_loss += loss.item() * len(batch_lstm)
        
        val_loss /= len(val_lstm)
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

def evaluate_hybrid_model(model, lstm_data, gnn_sequences, targets):
    """Evaluate the hybrid model on test data.
    
    Args:
        model: Trained hybrid model
        lstm_data: LSTM input data
        gnn_sequences: List of graph sequences for GNN
        targets: Target ball coordinates
        
    Returns:
        Test loss and predictions
    """
    # Convert LSTM data and targets to PyTorch tensors
    lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate
    criterion = nn.MSELoss()
    test_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for i in range(len(lstm_tensor)):
            # Forward pass
            prediction = model(lstm_tensor[i:i+1], gnn_sequences[i])
            predictions.append(prediction)
            
            # Compute loss
            loss = criterion(prediction, targets_tensor[i:i+1])
            test_loss += loss.item()
    
    test_loss /= len(lstm_tensor)
    predictions = torch.cat(predictions, dim=0).numpy()
    
    print(f'Test Loss: {test_loss:.6f}')
    
    return test_loss, predictions

def plot_hybrid_training_history(history):
    """Plot training and validation loss history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Hybrid LSTM-GNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('hybrid_training_history.png')
    plt.close()

def predict_with_hybrid(model, lstm_input, graph_sequence):
    """Make predictions with the trained hybrid model.
    
    Args:
        model: Trained hybrid model
        lstm_input: Input tensor for LSTM
        graph_sequence: Sequence of graphs for GNN
        
    Returns:
        Predicted ball coordinates
    """
    # Convert LSTM input to tensor
    lstm_tensor = torch.tensor(lstm_input, dtype=torch.float32)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(lstm_tensor, graph_sequence)
    
    return prediction.numpy()

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

def visualize_hybrid_predictions(actual, lstm_pred, gnn_pred, hybrid_pred, home_players, away_players, start_idx, end_idx):
    """Visualize actual vs predicted ball positions from different models.
    
    Args:
        actual: Actual ball positions
        lstm_pred: LSTM model predictions
        gnn_pred: GNN model predictions
        hybrid_pred: Hybrid model predictions
        home_players: DataFrame with home player positions
        away_players: DataFrame with away player positions
        start_idx: Start index for visualization
        end_idx: End index for visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot field boundaries (assuming standard dimensions)
    field_length = 105
    field_width = 68
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Draw penalty areas
    plt.plot([0, 16.5, 16.5, 0], [field_width/2 - 20.15, field_width/2 - 20.15, field_width/2 + 20.15, field_width/2 + 20.15], 'k-')
    plt.plot([field_length, field_length - 16.5, field_length - 16.5, field_length], 
             [field_width/2 - 20.15, field_width/2 - 20.15, field_width/2 + 20.15, field_width/2 + 20.15], 'k-')
    
    # Plot actual ball trajectory
    actual_x = actual[start_idx:end_idx, 0]
    actual_y = actual[start_idx:end_idx, 1]
    plt.plot(actual_x, actual_y, 'k-', label='Actual Ball Path', linewidth=2)
    
    # Plot LSTM predicted trajectory
    lstm_x = lstm_pred[start_idx:end_idx, 0]
    lstm_y = lstm_pred[start_idx:end_idx, 1]
    plt.plot(lstm_x, lstm_y, 'b--', label='LSTM Prediction', linewidth=2)
    
    # Plot GNN predicted trajectory
    gnn_x = gnn_pred[start_idx:end_idx, 0]
    gnn_y = gnn_pred[start_idx:end_idx, 1]
    plt.plot(gnn_x, gnn_y, 'g--', label='GNN Prediction', linewidth=2)
    
    # Plot hybrid predicted trajectory
    hybrid_x = hybrid_pred[start_idx:end_idx, 0]
    hybrid_y = hybrid_pred[start_idx:end_idx, 1]
    plt.plot(hybrid_x, hybrid_y, 'r-', label='Hybrid Prediction', linewidth=2)
    
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
    plt.plot(actual_x[-1], actual_y[-1], 'ko', markersize=10, label='Actual Ball')
    
    # Plot the predicted ball positions for the last frame
    plt.plot(lstm_x[-1], lstm_y[-1], 'bo', markersize=8, label='LSTM Prediction')
    plt.plot(gnn_x[-1], gnn_y[-1], 'go', markersize=8, label='GNN Prediction')
    plt.plot(hybrid_x[-1], hybrid_y[-1], 'ro', markersize=8, label='Hybrid Prediction')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Comparison of Ball Position Predictions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('hybrid_prediction_comparison.png')
    plt.close()

def compare_model_performance(actual, lstm_pred, gnn_pred, hybrid_pred):
    """Compare performance metrics of different models.
    
    Args:
        actual: Actual ball positions
        lstm_pred: LSTM model predictions
        gnn_pred: GNN model predictions
        hybrid_pred: Hybrid model predictions
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate Mean Squared Error (MSE)
    lstm_mse = np.mean(np.sum((actual - lstm_pred)**2, axis=1))
    gnn_mse = np.mean(np.sum((actual - gnn_pred)**2, axis=1))
    hybrid_mse = np.mean(np.sum((actual - hybrid_pred)**2, axis=1))
    
    # Calculate Mean Absolute Error (MAE)
    lstm_mae = np.mean(np.sum(np.abs(actual - lstm_pred), axis=1))
    gnn_mae = np.mean(np.sum(np.abs(actual - gnn_pred), axis=1))
    hybrid_mae = np.mean(np.sum(np.abs(actual - hybrid_pred), axis=1))
    
    # Calculate Root Mean Squared Error (RMSE)
    lstm_rmse = np.sqrt(lstm_mse)
    gnn_rmse = np.sqrt(gnn_mse)
    hybrid_rmse = np.sqrt(hybrid_mse)
    
    # Print results
    print("\nModel Performance Comparison:")
    print(f"{'Model':<15} {'MSE':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 45)
    print(f"{'LSTM':<15} {lstm_mse:<10.4f} {lstm_mae:<10.4f} {lstm_rmse:<10.4f}")
    print(f"{'GNN':<15} {gnn_mse:<10.4f} {gnn_mae:<10.4f} {gnn_rmse:<10.4f}")
    print(f"{'Hybrid':<15} {hybrid_mse:<10.4f} {hybrid_mae:<10.4f} {hybrid_rmse:<10.4f}")
    
    # Calculate improvement percentages
    lstm_improvement = (lstm_mse - hybrid_mse) / lstm_mse * 100
    gnn_improvement = (gnn_mse - hybrid_mse) / gnn_mse * 100
    
    print(f"\nHybrid model improves over LSTM by {lstm_improvement:.2f}%")
    print(f"Hybrid model improves over GNN by {gnn_improvement:.2f}%")
    
    # Return metrics
    return {
        'lstm': {'mse': lstm_mse, 'mae': lstm_mae, 'rmse': lstm_rmse},
        'gnn': {'mse': gnn_mse, 'mae': gnn_mae, 'rmse': gnn_rmse},
        'hybrid': {'mse': hybrid_mse, 'mae': hybrid_mae, 'rmse': hybrid_rmse},
        'improvement': {'over_lstm': lstm_improvement, 'over_gnn': gnn_improvement}
    }

def plot_performance_comparison(metrics):
    """Plot performance comparison of different models.
    
    Args:
        metrics: Dictionary of performance metrics
    """
    # Extract metrics
    models = ['LSTM', 'GNN', 'Hybrid']
    mse_values = [metrics['lstm']['mse'], metrics['gnn']['mse'], metrics['hybrid']['mse']]
    mae_values = [metrics['lstm']['mae'], metrics['gnn']['mae'], metrics['hybrid']['mae']]
    rmse_values = [metrics['lstm']['rmse'], metrics['gnn']['rmse'], metrics['hybrid']['rmse']]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot MSE
    axes[0].bar(models, mse_values, color=['blue', 'green', 'red'])
    axes[0].set_title('Mean Squared Error (MSE)')
    axes[0].set_ylabel('MSE')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot MAE
    axes[1].bar(models, mae_values, color=['blue', 'green', 'red'])
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_ylabel('MAE')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot RMSE
    axes[2].bar(models, rmse_values, color=['blue', 'green', 'red'])
    axes[2].set_title('Root Mean Squared Error (RMSE)')
    axes[2].set_ylabel('RMSE')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()
    
    # Create improvement plot
    plt.figure(figsize=(10, 6))
    improvements = [metrics['improvement']['over_lstm'], metrics['improvement']['over_gnn']]
    plt.bar(['Improvement over LSTM', 'Improvement over GNN'], improvements, color=['skyblue', 'lightgreen'])
    plt.title('Hybrid Model Improvement (%)')
    plt.ylabel('Improvement (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('hybrid_model_improvement.png')
    plt.close()
