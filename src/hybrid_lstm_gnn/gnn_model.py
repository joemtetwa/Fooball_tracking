import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import networkx as nx
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

from src.hybrid_lstm_gnn.data_utils import create_graph

class BallPredictorGNN(nn.Module):
    """Graph Neural Network model for predicting ball coordinates."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, use_attention=True):
        """Initialize the GNN model.
        
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Output dimension (2 for x,y coordinates)
            use_attention: Whether to use Graph Attention instead of Graph Convolution
        """
        super().__init__()
        self.use_attention = use_attention
        
        # First graph layer
        if use_attention:
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.2)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2)
        else:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, data):
        """Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing graph information
            
        Returns:
            Tensor with predicted ball coordinates
        """
        x, edge_index = data.x, data.edge_index
        
        # First graph convolution/attention layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second graph convolution/attention layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Get the ball node features (assuming it's the last node)
        ball_features = x[-1]
        
        # Fully connected layers
        x = self.fc1(ball_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TemporalGNN(nn.Module):
    """Temporal Graph Neural Network for sequence of graphs."""
    
    def __init__(self, input_dim, hidden_dim=64, lstm_hidden_dim=128, output_dim=2, use_attention=True):
        """Initialize the Temporal GNN model.
        
        Args:
            input_dim: Number of node features
            hidden_dim: Size of GNN hidden layers
            lstm_hidden_dim: Size of LSTM hidden layers
            output_dim: Output dimension (2 for x,y coordinates)
            use_attention: Whether to use Graph Attention instead of Graph Convolution
        """
        super().__init__()
        
        # GNN for processing each graph in the sequence
        self.gnn = BallPredictorGNN(input_dim, hidden_dim, hidden_dim, use_attention)
        
        # LSTM for processing the sequence of GNN outputs
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # Fully connected layers for final prediction
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_dim // 2, output_dim)
        )
    
    def forward(self, data_list):
        """Forward pass through the network.
        
        Args:
            data_list: List of PyTorch Geometric Data objects representing a sequence of graphs
            
        Returns:
            Tensor with predicted ball coordinates
        """
        # Process each graph in the sequence with the GNN
        gnn_outputs = []
        for data in data_list:
            # Get GNN output (ball node features)
            gnn_out = self.gnn(data)
            gnn_outputs.append(gnn_out)
        
        # Stack GNN outputs to form a sequence
        gnn_sequence = torch.stack(gnn_outputs, dim=0)
        
        # Add batch dimension if processing a single sequence
        if len(gnn_sequence.shape) == 2:
            gnn_sequence = gnn_sequence.unsqueeze(0)
        
        # Process the sequence with LSTM
        lstm_out, _ = self.lstm(gnn_sequence)
        
        # Use the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        prediction = self.fc(last_output)
        
        return prediction

def train_gnn_model(graph_sequences, targets, batch_size=32, epochs=50, learning_rate=0.001, model_save_path=None):
    """Train the Temporal GNN model.
    
    Args:
        graph_sequences: List of graph sequences (each sequence is a list of PyTorch Geometric Data objects)
        targets: Target ball coordinates
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the trained model
        
    Returns:
        Trained model and training history
    """
    # Determine input dimension from the first graph
    input_dim = graph_sequences[0][0].x.shape[1]
    
    # Initialize model
    model = TemporalGNN(input_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Split data into training and validation sets (80/20 split)
    n_samples = len(graph_sequences)
    n_train = int(0.8 * n_samples)
    
    train_sequences = graph_sequences[:n_train]
    train_targets = targets_tensor[:n_train]
    
    val_sequences = graph_sequences[n_train:]
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
        for i in range(0, len(train_sequences), batch_size):
            batch_sequences = train_sequences[i:i + batch_size]
            batch_targets = train_targets[i:i + batch_size]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass (process each sequence individually)
            batch_predictions = []
            for sequence in batch_sequences:
                prediction = model(sequence)
                batch_predictions.append(prediction)
            
            # Stack predictions
            batch_predictions = torch.cat(batch_predictions, dim=0)
            
            # Compute loss
            loss = criterion(batch_predictions, batch_targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(batch_sequences)
        
        train_loss /= len(train_sequences)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch_sequences = val_sequences[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]
                
                # Forward pass (process each sequence individually)
                batch_predictions = []
                for sequence in batch_sequences:
                    prediction = model(sequence)
                    batch_predictions.append(prediction)
                
                # Stack predictions
                batch_predictions = torch.cat(batch_predictions, dim=0)
                
                # Compute loss
                loss = criterion(batch_predictions, batch_targets)
                val_loss += loss.item() * len(batch_sequences)
        
        val_loss /= len(val_sequences)
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

def evaluate_gnn_model(model, graph_sequences, targets):
    """Evaluate the Temporal GNN model on test data.
    
    Args:
        model: Trained Temporal GNN model
        graph_sequences: List of graph sequences for testing
        targets: Target ball coordinates
        
    Returns:
        Test loss and predictions
    """
    # Convert targets to tensor
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate
    criterion = nn.MSELoss()
    test_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for i, sequence in enumerate(graph_sequences):
            # Forward pass
            prediction = model(sequence)
            predictions.append(prediction)
            
            # Compute loss
            loss = criterion(prediction, targets_tensor[i:i+1])
            test_loss += loss.item()
    
    test_loss /= len(graph_sequences)
    predictions = torch.cat(predictions, dim=0).numpy()
    
    print(f'Test Loss: {test_loss:.6f}')
    
    return test_loss, predictions

def plot_gnn_training_history(history):
    """Plot training and validation loss history.
    
    Args:
        history: Dictionary containing training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('gnn_training_history.png')
    plt.close()

def predict_with_gnn(model, graph_sequence):
    """Make predictions with the trained Temporal GNN model.
    
    Args:
        model: Trained Temporal GNN model
        graph_sequence: Sequence of graphs for prediction
        
    Returns:
        Predicted ball coordinates
    """
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction = model(graph_sequence)
    
    return prediction.numpy()

def visualize_gnn_graph(graph, title="Graph Structure"):
    """Visualize the graph structure.
    
    Args:
        graph: PyTorch Geometric Data object
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Extract node positions and features
    node_pos = graph.x[:, :2].numpy()  # x, y coordinates
    node_types = graph.x[:, 4].numpy()  # team type (0=home, 1=away, 2=ball)
    
    # Extract edges
    edges = graph.edge_index.numpy()
    
    # Plot nodes
    for i in range(len(node_pos)):
        if node_types[i] == 0:  # Home team
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'bo', markersize=8)
        elif node_types[i] == 1:  # Away team
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'ro', markersize=8)
        else:  # Ball
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'ko', markersize=10)
    
    # Plot edges
    for j in range(edges.shape[1]):
        src, dst = edges[0, j], edges[1, j]
        plt.plot([node_pos[src, 0], node_pos[dst, 0]], 
                 [node_pos[src, 1], node_pos[dst, 1]], 'k-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('gnn_graph_visualization.png')
    plt.close()

def visualize_gnn_predictions(actual, predicted, graph_sequences, start_idx, end_idx):
    """Visualize actual vs predicted ball positions with graph structure.
    
    Args:
        actual: Actual ball positions
        predicted: Predicted ball positions
        graph_sequences: List of graph sequences
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
    
    # Plot actual ball trajectory
    actual_x = actual[start_idx:end_idx, 0]
    actual_y = actual[start_idx:end_idx, 1]
    plt.plot(actual_x, actual_y, 'b-', label='Actual Ball Path', linewidth=2)
    
    # Plot predicted ball trajectory
    pred_x = predicted[start_idx:end_idx, 0]
    pred_y = predicted[start_idx:end_idx, 1]
    plt.plot(pred_x, pred_y, 'r--', label='Predicted Ball Path', linewidth=2)
    
    # Plot the graph structure for the last frame
    last_graph = graph_sequences[end_idx-1][-1]  # Last graph in the sequence
    
    # Extract node positions and features
    node_pos = last_graph.x[:, :2].numpy()  # x, y coordinates
    node_types = last_graph.x[:, 4].numpy()  # team type (0=home, 1=away, 2=ball)
    
    # Extract edges
    edges = last_graph.edge_index.numpy()
    
    # Plot nodes
    for i in range(len(node_pos)):
        if node_types[i] == 0:  # Home team
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'bo', markersize=8)
        elif node_types[i] == 1:  # Away team
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'ro', markersize=8)
        else:  # Ball
            plt.plot(node_pos[i, 0], node_pos[i, 1], 'ko', markersize=10)
    
    # Plot edges
    for j in range(edges.shape[1]):
        src, dst = edges[0, j], edges[1, j]
        plt.plot([node_pos[src, 0], node_pos[dst, 0]], 
                 [node_pos[src, 1], node_pos[dst, 1]], 'k-', alpha=0.3)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('GNN Ball Position Prediction with Graph Structure')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('gnn_prediction_visualization.png')
    plt.close()
