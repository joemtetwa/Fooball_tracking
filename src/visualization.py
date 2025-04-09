import matplotlib.pyplot as plt
import os
import numpy as np

def plot_ball_movement_analysis(predicted_positions, window_size=50, title="Ball Movement Analysis"):
    """Create a detailed analysis of ball movement patterns."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trajectory Plot with Direction Arrows
    ax1.plot(predicted_positions['ball_x_pred'], predicted_positions['ball_y_pred'],
            'b-', alpha=0.6, label='Ball Path')
    
    # Add direction arrows every window_size frames
    for i in range(0, len(predicted_positions), window_size):
        if i + 1 < len(predicted_positions):
            dx = predicted_positions['ball_x_pred'].iloc[i+1] - predicted_positions['ball_x_pred'].iloc[i]
            dy = predicted_positions['ball_y_pred'].iloc[i+1] - predicted_positions['ball_y_pred'].iloc[i]
            ax1.arrow(predicted_positions['ball_x_pred'].iloc[i], 
                     predicted_positions['ball_y_pred'].iloc[i],
                     dx, dy, head_width=20, head_length=30, fc='r', ec='r', alpha=0.5)
    
    ax1.set_title("Ball Trajectory with Movement Direction")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(True)
    
    # 2. Speed Plot
    speeds = np.sqrt(
        np.diff(predicted_positions['ball_x_pred'])**2 + 
        np.diff(predicted_positions['ball_y_pred'])**2
    )
    ax2.plot(predicted_positions['Time'].iloc[1:], speeds, 'g-', alpha=0.7)
    ax2.set_title("Ball Speed Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Speed (units/frame)")
    ax2.grid(True)
    
    # 3. Heatmap of Ball Positions
    heatmap, xedges, yedges = np.histogram2d(
        predicted_positions['ball_x_pred'],
        predicted_positions['ball_y_pred'],
        bins=30
    )
    ax3.imshow(heatmap.T, origin='lower', 
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='YlOrRd')
    ax3.set_title("Ball Position Heatmap")
    ax3.set_xlabel("X Position")
    ax3.set_ylabel("Y Position")
    
    # 4. Possession Timeline
    possession_home = (predicted_positions['possessing_team_pred'] == 'home').astype(int)
    ax4.plot(predicted_positions['Time'], possession_home, 'b-', alpha=0.7)
    ax4.set_title("Ball Possession Timeline")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Possession (0=Away, 1=Home)")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Away', 'Home'])
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def plot_ball_trajectory(true_positions=None, predicted_positions=None, title="Ball Trajectory"):
    """Plot the ball trajectory, showing both true and predicted positions if available."""
    plt.figure(figsize=(12, 8))
    
    if true_positions is not None and 'ball_x' in true_positions.columns:
        plt.plot(true_positions['ball_x'], true_positions['ball_y'],
                'b-', label='True Position', alpha=0.6)
    
    if predicted_positions is not None:
        plt.plot(predicted_positions['ball_x_pred'], predicted_positions['ball_y_pred'],
                'r--', label='Predicted Position', alpha=0.8)
    
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_possession_timeline(true_possession=None, predicted_possession=None, title="Possession Timeline"):
    """Plot the possession timeline, showing both true and predicted possession if available."""
    plt.figure(figsize=(15, 5))
    
    if true_possession is not None and 'possessing_team' in true_possession.columns:
        possession_values = (true_possession['possessing_team'] == 'home').astype(int)
        plt.plot(true_possession['Time'], possession_values,
                'b-', label='True Possession', alpha=0.6)
    
    if predicted_possession is not None:
        pred_possession_values = (predicted_possession['possessing_team_pred'] == 'home').astype(int)
        plt.plot(predicted_possession['Time'], pred_possession_values,
                'r--', label='Predicted Possession', alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Possession (0=Away, 1=Home)')
    plt.yticks([0, 1], ['Away', 'Home'])
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_feature_importance(importance_df, title="Feature Importance", top_n=20):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create bar plot
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return plt.gcf()

def save_all_plots(df_results, model=None):
    """Generate and save all visualization plots."""
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot ball movement analysis
    fig = plot_ball_movement_analysis(df_results)
    fig.savefig(os.path.join(plots_dir, 'ball_movement_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot ball trajectory
    fig = plot_ball_trajectory(predicted_positions=df_results)
    fig.savefig(os.path.join(plots_dir, 'ball_trajectory.png'))
    plt.close(fig)
    
    # Plot possession timeline
    fig = plot_possession_timeline(predicted_possession=df_results)
    fig.savefig(os.path.join(plots_dir, 'possession_timeline.png'))
    plt.close(fig)
    
    # Plot feature importance if model is provided
    if model is not None:
        ball_importance, possession_importance = model.get_feature_importance()
        
        fig = plot_feature_importance(ball_importance, 
                                    title="Ball Position Model - Feature Importance")
        fig.savefig(os.path.join(plots_dir, 'ball_feature_importance.png'))
        plt.close(fig)
        
        fig = plot_feature_importance(possession_importance, 
                                    title="Possession Model - Feature Importance")
        fig.savefig(os.path.join(plots_dir, 'possession_feature_importance.png'))
        plt.close(fig)
    
    print(f"All plots saved to {plots_dir}/")
