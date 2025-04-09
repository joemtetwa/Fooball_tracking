import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.gridspec as gridspec

def create_architecture_diagram(output_path):
    """
    Create a visual diagram of the enhanced hybrid LSTM-GNN architecture with player influence.
    
    Args:
        output_path: Path to save the diagram
    """
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Set up grid for layout
    gs = gridspec.GridSpec(4, 3, height_ratios=[1, 2, 2, 1])
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Colors
    lstm_color = '#4285F4'  # Blue
    gnn_color = '#EA4335'   # Red
    pass_color = '#FBBC05'  # Yellow
    fusion_color = '#34A853'  # Green
    influence_color = '#9C27B0'  # Purple
    
    # Input data section
    ax_input = plt.subplot(gs[0, :])
    ax_input.set_xlim(0, 10)
    ax_input.set_ylim(0, 1)
    ax_input.axis('off')
    
    input_rect = patches.Rectangle((0.5, 0.2), 9, 0.6, linewidth=2, edgecolor='k', facecolor='lightgray', alpha=0.5)
    ax_input.add_patch(input_rect)
    ax_input.text(5, 0.5, 'Input Data: Player Positions, Ball Coordinates (when available)', 
                 ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw arrow down
    ax_input.arrow(5, 0.2, 0, -0.1, head_width=0.1, head_length=0.05, fc='k', ec='k', linewidth=2)
    
    # LSTM Component
    ax_lstm = plt.subplot(gs[1, 0])
    ax_lstm.set_xlim(0, 10)
    ax_lstm.set_ylim(0, 10)
    ax_lstm.axis('off')
    
    # Draw LSTM box
    lstm_rect = patches.Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='k', facecolor=lstm_color, alpha=0.3)
    ax_lstm.add_patch(lstm_rect)
    ax_lstm.text(5, 9, 'LSTM Component', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw LSTM cells
    for i in range(5):
        y_pos = 7 - i
        lstm_cell = patches.Rectangle((2, y_pos), 6, 0.8, linewidth=1, edgecolor='k', facecolor=lstm_color, alpha=0.7)
        ax_lstm.add_patch(lstm_cell)
        ax_lstm.text(5, y_pos + 0.4, f'LSTM Cell {i+1}', ha='center', va='center', fontsize=10)
    
    # Draw output arrow
    ax_lstm.arrow(9, 5, 1, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', linewidth=2)
    
    # GNN Component
    ax_gnn = plt.subplot(gs[1, 1])
    ax_gnn.set_xlim(0, 10)
    ax_gnn.set_ylim(0, 10)
    ax_gnn.axis('off')
    
    # Draw GNN box
    gnn_rect = patches.Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='k', facecolor=gnn_color, alpha=0.3)
    ax_gnn.add_patch(gnn_rect)
    ax_gnn.text(5, 9, 'GNN Component', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw nodes and edges for GNN
    # Ball node
    ball_node = plt.Circle((5, 5), 0.5, facecolor=gnn_color, edgecolor='k', alpha=0.8)
    ax_gnn.add_patch(ball_node)
    ax_gnn.text(5, 5, 'Ball', ha='center', va='center', fontsize=8)
    
    # Player nodes
    player_positions = [
        (3, 7), (7, 7),  # Top players
        (2, 5), (8, 5),  # Middle players
        (3, 3), (7, 3)   # Bottom players
    ]
    
    for i, (x, y) in enumerate(player_positions):
        team = 'H' if i < 3 else 'A'
        player_node = plt.Circle((x, y), 0.4, facecolor='blue' if team == 'H' else 'red', edgecolor='k', alpha=0.8)
        ax_gnn.add_patch(player_node)
        ax_gnn.text(x, y, f'{team}{i%3+1}', ha='center', va='center', fontsize=8, color='white')
        
        # Draw edge to ball
        ax_gnn.plot([x, 5], [y, 5], 'k-', alpha=0.5)
    
    # Draw output arrow
    ax_gnn.arrow(9, 5, 1, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', linewidth=2)
    
    # Pass Analysis Component
    ax_pass = plt.subplot(gs[1, 2])
    ax_pass.set_xlim(0, 10)
    ax_pass.set_ylim(0, 10)
    ax_pass.axis('off')
    
    # Draw Pass Analysis box
    pass_rect = patches.Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='k', facecolor=pass_color, alpha=0.3)
    ax_pass.add_patch(pass_rect)
    ax_pass.text(5, 9, 'Pass Analysis Component', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw pass matrix visualization
    matrix_rect = patches.Rectangle((2, 3), 6, 4, linewidth=1, edgecolor='k', facecolor=pass_color, alpha=0.7)
    ax_pass.add_patch(matrix_rect)
    
    # Draw matrix grid
    for i in range(4):
        for j in range(4):
            cell_rect = patches.Rectangle((3+j, 4+i), 0.8, 0.8, linewidth=1, edgecolor='k', facecolor='white', alpha=0.7)
            ax_pass.add_patch(cell_rect)
            # Add some random values to represent pass probabilities
            if i != j:  # No self-passes
                value = np.random.rand() * 0.8
                ax_pass.text(3.4+j, 4.4+i, f'{value:.1f}', ha='center', va='center', fontsize=8)
    
    ax_pass.text(5, 2.5, 'Pass Probability Matrix', ha='center', va='center', fontsize=10)
    
    # Draw output arrow
    ax_pass.arrow(5, 1, 0, -1, head_width=0.3, head_length=0.3, fc='k', ec='k', linewidth=2)
    
    # Fusion Layer
    ax_fusion = plt.subplot(gs[2, 0:2])
    ax_fusion.set_xlim(0, 20)
    ax_fusion.set_ylim(0, 10)
    ax_fusion.axis('off')
    
    # Draw Fusion box
    fusion_rect = patches.Rectangle((1, 1), 18, 8, linewidth=2, edgecolor='k', facecolor=fusion_color, alpha=0.3)
    ax_fusion.add_patch(fusion_rect)
    ax_fusion.text(10, 9, 'Fusion Layer', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw fusion components
    lstm_out = patches.Rectangle((2, 6), 4, 1, linewidth=1, edgecolor='k', facecolor=lstm_color, alpha=0.7)
    ax_fusion.add_patch(lstm_out)
    ax_fusion.text(4, 6.5, 'LSTM Output', ha='center', va='center', fontsize=10)
    
    gnn_out = patches.Rectangle((8, 6), 4, 1, linewidth=1, edgecolor='k', facecolor=gnn_color, alpha=0.7)
    ax_fusion.add_patch(gnn_out)
    ax_fusion.text(10, 6.5, 'GNN Output', ha='center', va='center', fontsize=10)
    
    pass_out = patches.Rectangle((14, 6), 4, 1, linewidth=1, edgecolor='k', facecolor=pass_color, alpha=0.7)
    ax_fusion.add_patch(pass_out)
    ax_fusion.text(16, 6.5, 'Pass Features', ha='center', va='center', fontsize=10)
    
    # Draw concatenation
    concat_rect = patches.Rectangle((2, 4), 16, 1, linewidth=1, edgecolor='k', facecolor=fusion_color, alpha=0.7)
    ax_fusion.add_patch(concat_rect)
    ax_fusion.text(10, 4.5, 'Concatenated Features', ha='center', va='center', fontsize=10)
    
    # Draw arrows to concatenation
    ax_fusion.arrow(4, 6, 0, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax_fusion.arrow(10, 6, 0, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax_fusion.arrow(16, 6, 0, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    # Draw fully connected layers
    fc1_rect = patches.Rectangle((6, 2.5), 8, 0.8, linewidth=1, edgecolor='k', facecolor=fusion_color, alpha=0.7)
    ax_fusion.add_patch(fc1_rect)
    ax_fusion.text(10, 2.9, 'Fully Connected Layer', ha='center', va='center', fontsize=10)
    
    # Draw arrow from concatenation to FC
    ax_fusion.arrow(10, 4, 0, -0.7, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    # Draw output
    output_rect = patches.Rectangle((8, 1), 4, 0.8, linewidth=1, edgecolor='k', facecolor=fusion_color, alpha=0.9)
    ax_fusion.add_patch(output_rect)
    ax_fusion.text(10, 1.4, 'Ball Coordinates', ha='center', va='center', fontsize=10)
    
    # Draw arrow from FC to output
    ax_fusion.arrow(10, 2.5, 0, -0.7, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    # Draw output arrow to player influence
    ax_fusion.arrow(12, 1.4, 8, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', linewidth=2)
    
    # Player Influence Module
    ax_influence = plt.subplot(gs[2, 2])
    ax_influence.set_xlim(0, 10)
    ax_influence.set_ylim(0, 10)
    ax_influence.axis('off')
    
    # Draw Influence box
    influence_rect = patches.Rectangle((1, 1), 8, 8, linewidth=2, edgecolor='k', facecolor=influence_color, alpha=0.3)
    ax_influence.add_patch(influence_rect)
    ax_influence.text(5, 9, 'Player Influence Module', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw field with players and influence zones
    field_rect = patches.Rectangle((2, 2), 6, 6, linewidth=1, edgecolor='k', facecolor='lightgreen', alpha=0.3)
    ax_influence.add_patch(field_rect)
    
    # Draw ball
    ball = plt.Circle((5, 5), 0.3, facecolor='yellow', edgecolor='k', alpha=0.8)
    ax_influence.add_patch(ball)
    
    # Draw players with influence zones
    player_positions = [
        (3, 6, 'blue'), (7, 6, 'red'),
        (4, 4, 'blue'), (6, 4, 'red'),
        (3, 3, 'blue'), (7, 3, 'red')
    ]
    
    for x, y, color in player_positions:
        # Influence zone
        influence_zone = plt.Circle((x, y), 1.5, facecolor=color, edgecolor='k', alpha=0.1)
        ax_influence.add_patch(influence_zone)
        
        # Player
        player = plt.Circle((x, y), 0.3, facecolor=color, edgecolor='k', alpha=0.8)
        ax_influence.add_patch(player)
    
    # Draw output arrow
    ax_influence.arrow(5, 1, 0, -1, head_width=0.3, head_length=0.3, fc='k', ec='k', linewidth=2)
    
    # Final Output
    ax_output = plt.subplot(gs[3, :])
    ax_output.set_xlim(0, 10)
    ax_output.set_ylim(0, 1)
    ax_output.axis('off')
    
    output_rect = patches.Rectangle((0.5, 0.2), 9, 0.6, linewidth=2, edgecolor='k', facecolor='lightgray', alpha=0.5)
    ax_output.add_patch(output_rect)
    ax_output.text(5, 0.5, 'Enhanced Ball Coordinate Predictions with Player Influence', 
                 ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Title
    plt.suptitle('Enhanced Hybrid LSTM-GNN Architecture with Player Influence', fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved architecture diagram to {output_path}")

if __name__ == "__main__":
    output_dir = "visualizations/presentation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "architecture_diagram.png")
    create_architecture_diagram(output_path)
