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

class EnhancedHybridModel(nn.Module):
    """
    Enhanced hybrid model that incorporates pass analysis features.
    """
    
    def __init__(self, lstm_input_size, gnn_input_dim, pass_feature_size=10, 
                 lstm_hidden_size=128, gnn_hidden_dim=64, fusion_hidden_size=64):
        """
        Initialize the enhanced hybrid model.
        
        Args:
            lstm_input_size: Number of features for LSTM input
            gnn_input_dim: Number of node features for GNN
            pass_feature_size: Number of pass analysis features
            lstm_hidden_size: Size of LSTM hidden layers
            gnn_hidden_dim: Size of GNN hidden layers
            fusion_hidden_size: Size of fusion layer
        """
        super().__init__()
        
        # LSTM component
        self.lstm = BallPredictorLSTM(lstm_input_size, lstm_hidden_size)
        
        # GNN component
        self.gnn = TemporalGNN(gnn_input_dim, gnn_hidden_dim, lstm_hidden_size, 
                              gnn_hidden_dim, use_attention=True)
        
        # Pass analysis component
        self.pass_network = nn.Sequential(
            nn.Linear(pass_feature_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        # Store the fusion hidden size for later use
        self.fusion_hidden_size = fusion_hidden_size
        
        # We'll create the fusion layer in the forward pass after determining the actual input size
    
    def forward(self, lstm_input, graph_sequences, pass_features):
        """
        Forward pass through the network.
        
        Args:
            lstm_input: Input tensor for LSTM
            graph_sequences: Batch of graph sequences for GNN
            pass_features: Pass analysis features
            
        Returns:
            Predicted ball coordinates
        """
        # Get LSTM prediction - shape: [batch_size, 2]  
        lstm_output = self.lstm(lstm_input)
        
        # Process each graph sequence separately and collect outputs
        batch_size = lstm_input.shape[0]
        gnn_outputs = []
        
        for i in range(batch_size):
            # Process a single graph sequence
            if i < len(graph_sequences):
                # Get GNN prediction for this sequence - shape: [1, 2]
                single_output = self.gnn(graph_sequences[i])
                gnn_outputs.append(single_output)
            else:
                # Handle case where there are fewer graph sequences than batch size
                # Use zeros as a fallback
                gnn_outputs.append(torch.zeros(2, device=lstm_output.device))
        
        # Stack GNN outputs into a batch tensor - shape: [batch_size, 2]
        gnn_output = torch.stack(gnn_outputs)
        
        # Get pass network prediction - shape: [batch_size, 2]
        pass_output = self.pass_network(pass_features)
        
        # Make sure all outputs have the same shape [batch_size, 2]
        if len(lstm_output.shape) > 2:
            lstm_output = lstm_output.squeeze(1)
        if len(gnn_output.shape) > 2:
            gnn_output = gnn_output.squeeze(1)
        if len(pass_output.shape) > 2:
            pass_output = pass_output.squeeze(1)
            
        # Concatenate all outputs
        combined = torch.cat((lstm_output, gnn_output, pass_output), dim=1)
        
        # Create the fusion layer dynamically if it doesn't exist yet
        if not hasattr(self, 'fusion') or self.fusion[0].in_features != combined.shape[1]:
            fusion_input_size = combined.shape[1]
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_size, self.fusion_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.fusion_hidden_size, 2)  # Final output: x, y coordinates
            )
            # Move to the same device as the input
            self.fusion = self.fusion.to(combined.device)
        
        # Final prediction through fusion layer
        prediction = self.fusion(combined)
        
        return prediction


def train_enhanced_hybrid_model(lstm_data, gnn_sequences, pass_features, targets, 
                               batch_size=32, epochs=50, learning_rate=0.001, model_save_path=None):
    """
    Train the enhanced hybrid model.
    
    Args:
        lstm_data: LSTM input data
        gnn_sequences: List of graph sequences for GNN
        pass_features: Pass analysis features
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
    pass_feature_size = pass_features.shape[1]
    
    # Initialize model
    model = EnhancedHybridModel(lstm_input_size, gnn_input_dim, pass_feature_size)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to PyTorch tensors
    lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    pass_tensor = torch.tensor(pass_features, dtype=torch.float32)
    
    # Split data into training and validation sets (80/20 split)
    n_samples = len(lstm_tensor)
    n_train = int(0.8 * n_samples)
    
    train_lstm = lstm_tensor[:n_train]
    train_gnn = gnn_sequences[:n_train]
    train_pass = pass_tensor[:n_train]
    train_targets = targets_tensor[:n_train]
    
    val_lstm = lstm_tensor[n_train:]
    val_gnn = gnn_sequences[n_train:]
    val_pass = pass_tensor[n_train:]
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
            # Get batch
            batch_lstm = train_lstm[i:i+batch_size]
            batch_gnn = train_gnn[i:i+batch_size]
            batch_pass = train_pass[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_lstm, batch_gnn, batch_pass)
            
            # Compute loss
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_lstm)
        
        train_loss /= len(train_lstm)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_lstm), batch_size):
                # Get batch
                batch_lstm = val_lstm[i:i+batch_size]
                batch_gnn = val_gnn[i:i+batch_size]
                batch_pass = val_pass[i:i+batch_size]
                batch_targets = val_targets[i:i+batch_size]
                
                # Forward pass
                outputs = model(batch_lstm, batch_gnn, batch_pass)
                
                # Compute loss
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item() * len(batch_lstm)
            
            val_loss /= len(val_lstm)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss and model_save_path:
            best_val_loss = val_loss
            torch.save(model, model_save_path)
            print(f"Saved best model with validation loss: {val_loss:.6f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # If model wasn't saved during training, save final model
    if model_save_path and not os.path.exists(model_save_path):
        torch.save(model, model_save_path)
        print(f"Saved final model to {model_save_path}")
    
    return model, history


def evaluate_enhanced_hybrid_model(model, lstm_data, gnn_sequences, pass_features, targets):
    """
    Evaluate the enhanced hybrid model on test data.
    
    Args:
        model: Trained enhanced hybrid model
        lstm_data: LSTM input data
        gnn_sequences: List of graph sequences for GNN
        pass_features: Pass analysis features
        targets: Target ball coordinates
        
    Returns:
        Test loss and predictions
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert data to PyTorch tensors
    lstm_tensor = torch.tensor(lstm_data, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    pass_tensor = torch.tensor(pass_features, dtype=torch.float32)
    
    # Evaluate model
    criterion = nn.MSELoss()
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        total_loss = 0.0
        
        for i in range(0, len(lstm_tensor), batch_size):
            # Get batch
            batch_lstm = lstm_tensor[i:i+batch_size]
            batch_gnn = gnn_sequences[i:i+batch_size]
            batch_pass = pass_tensor[i:i+batch_size]
            batch_targets = targets_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_lstm, batch_gnn, batch_pass)
            
            # Compute loss
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * len(batch_lstm)
            
            # Store predictions
            predictions.append(outputs.numpy())
    
    # Combine predictions
    predictions = np.vstack(predictions)
    
    # Calculate average loss
    avg_loss = total_loss / len(lstm_tensor)
    print(f"Test Loss: {avg_loss:.6f}")
    
    return avg_loss, predictions


def plot_enhanced_training_history(history):
    """
    Plot training and validation loss history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Enhanced Hybrid Model Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig("enhanced_training_history.png")
    plt.close()
    
    print("Training history plot saved to enhanced_training_history.png")


def visualize_enhanced_predictions(actual, lstm_pred, gnn_pred, pass_pred, enhanced_pred, 
                                  home_df, away_df, start_idx, end_idx):
    """
    Visualize actual vs predicted ball positions from different models.
    
    Args:
        actual: Actual ball positions
        lstm_pred: LSTM model predictions
        gnn_pred: GNN model predictions
        pass_pred: Pass analysis model predictions
        enhanced_pred: Enhanced hybrid model predictions
        home_df, away_df: DataFrames with player positions
        start_idx, end_idx: Start and end indices for visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Plot field boundaries (assuming standard dimensions)
    field_length = 105
    field_width = 68
    
    # Draw the field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Plot actual ball trajectory
    actual_x = actual[start_idx:end_idx, 0]
    actual_y = actual[start_idx:end_idx, 1]
    plt.plot(actual_x, actual_y, 'k-', label='Actual Ball Path', linewidth=2)
    
    # Plot LSTM predictions
    lstm_x = lstm_pred[start_idx:end_idx, 0]
    lstm_y = lstm_pred[start_idx:end_idx, 1]
    plt.plot(lstm_x, lstm_y, 'b--', label='LSTM Predictions', linewidth=1.5)
    
    # Plot GNN predictions
    gnn_x = gnn_pred[start_idx:end_idx, 0]
    gnn_y = gnn_pred[start_idx:end_idx, 1]
    plt.plot(gnn_x, gnn_y, 'g--', label='GNN Predictions', linewidth=1.5)
    
    # Plot Pass model predictions
    pass_x = pass_pred[start_idx:end_idx, 0]
    pass_y = pass_pred[start_idx:end_idx, 1]
    plt.plot(pass_x, pass_y, 'y--', label='Pass Model Predictions', linewidth=1.5)
    
    # Plot Enhanced hybrid predictions
    enhanced_x = enhanced_pred[start_idx:end_idx, 0]
    enhanced_y = enhanced_pred[start_idx:end_idx, 1]
    plt.plot(enhanced_x, enhanced_y, 'r-', label='Enhanced Hybrid Predictions', linewidth=2)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Enhanced Hybrid Model Ball Position Predictions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('enhanced_prediction_comparison.png')
    plt.close()
    
    print("Enhanced prediction visualization saved to enhanced_prediction_comparison.png")


def compare_enhanced_model_performance(actual, lstm_pred, gnn_pred, pass_pred, enhanced_pred):
    """
    Compare performance metrics of different models including the enhanced hybrid model.
    
    Args:
        actual: Actual ball positions
        lstm_pred: LSTM model predictions
        gnn_pred: GNN model predictions
        pass_pred: Pass analysis model predictions
        enhanced_pred: Enhanced hybrid model predictions
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate Mean Squared Error (MSE)
    lstm_mse = np.mean(np.sum((actual - lstm_pred)**2, axis=1))
    gnn_mse = np.mean(np.sum((actual - gnn_pred)**2, axis=1))
    pass_mse = np.mean(np.sum((actual - pass_pred)**2, axis=1))
    enhanced_mse = np.mean(np.sum((actual - enhanced_pred)**2, axis=1))
    
    # Calculate Mean Absolute Error (MAE)
    lstm_mae = np.mean(np.sum(np.abs(actual - lstm_pred), axis=1))
    gnn_mae = np.mean(np.sum(np.abs(actual - gnn_pred), axis=1))
    pass_mae = np.mean(np.sum(np.abs(actual - pass_pred), axis=1))
    enhanced_mae = np.mean(np.sum(np.abs(actual - enhanced_pred), axis=1))
    
    # Calculate Root Mean Squared Error (RMSE)
    lstm_rmse = np.sqrt(lstm_mse)
    gnn_rmse = np.sqrt(gnn_mse)
    pass_rmse = np.sqrt(pass_mse)
    enhanced_rmse = np.sqrt(enhanced_mse)
    
    # Print results
    print("\nModel Performance Comparison:")
    print(f"{'Model':<15} {'MSE':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 45)
    print(f"{'LSTM':<15} {lstm_mse:<10.4f} {lstm_mae:<10.4f} {lstm_rmse:<10.4f}")
    print(f"{'GNN':<15} {gnn_mse:<10.4f} {gnn_mae:<10.4f} {gnn_rmse:<10.4f}")
    print(f"{'Pass Model':<15} {pass_mse:<10.4f} {pass_mae:<10.4f} {pass_rmse:<10.4f}")
    print(f"{'Enhanced':<15} {enhanced_mse:<10.4f} {enhanced_mae:<10.4f} {enhanced_rmse:<10.4f}")
    
    # Calculate improvement percentages
    lstm_improvement = (lstm_mse - enhanced_mse) / lstm_mse * 100
    gnn_improvement = (gnn_mse - enhanced_mse) / gnn_mse * 100
    pass_improvement = (pass_mse - enhanced_mse) / pass_mse * 100
    
    print(f"\nEnhanced model improves over LSTM by {lstm_improvement:.2f}%")
    print(f"Enhanced model improves over GNN by {gnn_improvement:.2f}%")
    print(f"Enhanced model improves over Pass model by {pass_improvement:.2f}%")
    
    # Return metrics
    return {
        'lstm': {'mse': lstm_mse, 'mae': lstm_mae, 'rmse': lstm_rmse},
        'gnn': {'mse': gnn_mse, 'mae': gnn_mae, 'rmse': gnn_rmse},
        'pass': {'mse': pass_mse, 'mae': pass_mae, 'rmse': pass_rmse},
        'enhanced': {'mse': enhanced_mse, 'mae': enhanced_mae, 'rmse': enhanced_rmse},
        'improvement': {
            'over_lstm': lstm_improvement, 
            'over_gnn': gnn_improvement,
            'over_pass': pass_improvement
        }
    }
