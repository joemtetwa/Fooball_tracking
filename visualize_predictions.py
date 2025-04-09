import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize Ball Coordinate Predictions')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_ball_predictions.csv',
                        help='Path to the predictions CSV file')
    parser.add_argument('--match_num', type=int, default=4,
                        help='Match number to visualize')
    parser.add_argument('--start_frame', type=int, default=1000,
                        help='Starting frame for visualization')
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Number of frames to visualize')
    parser.add_argument('--save_animation', action='store_true',
                        help='Save animation as MP4 file')
    
    return parser.parse_args()

def load_data(match_num, prediction_mode=True):
    """Load match data."""
    from src.hybrid_lstm_gnn.data_utils import load_data
    return load_data(match_num, prediction_mode=prediction_mode)

def load_predictions(predictions_file):
    """Load predicted ball coordinates."""
    return pd.read_csv(predictions_file)

def create_animation(predictions, home_df, away_df, start_frame, num_frames, save_animation=False):
    """Create animation of predicted ball coordinates with player positions."""
    print("Creating animation...")
    
    # Set field dimensions
    field_length = 105  # meters
    field_width = 68    # meters
    
    # Get player position columns
    home_x_cols = [col for col in home_df.columns if '_x' in col and 'home_' in col]
    home_y_cols = [col for col in home_df.columns if '_y' in col and 'home_' in col]
    away_x_cols = [col for col in away_df.columns if '_x' in col and 'away_' in col]
    away_y_cols = [col for col in away_df.columns if '_y' in col and 'away_' in col]
    
    # Sort columns to ensure consistent order
    home_x_cols.sort()
    home_y_cols.sort()
    away_x_cols.sort()
    away_y_cols.sort()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Create slider for frame selection
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=start_frame,
        valmax=start_frame + num_frames - 1,
        valinit=start_frame,
        valstep=1
    )
    
    # Initialize plot elements
    field_outline, = ax.plot([], [], 'k-', linewidth=2)
    center_line, = ax.plot([], [], 'k-', linewidth=1)
    home_players = [ax.plot([], [], 'bo', markersize=8)[0] for _ in range(len(home_x_cols))]
    away_players = [ax.plot([], [], 'ro', markersize=8)[0] for _ in range(len(away_x_cols))]
    ball, = ax.plot([], [], 'yo', markersize=10)
    
    # Text elements for player labels
    home_labels = [ax.text(0, 0, '', fontsize=8, color='blue') for _ in range(len(home_x_cols))]
    away_labels = [ax.text(0, 0, '', fontsize=8, color='red') for _ in range(len(away_x_cols))]
    
    # Set field boundaries
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add title and labels
    ax.set_title('Ball Coordinate Predictions')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    
    def init():
        """Initialize animation."""
        # Draw field outline
        field_outline.set_data([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0])
        
        # Draw center line
        center_line.set_data([field_length/2, field_length/2], [0, field_width])
        
        # Draw center circle
        center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
        ax.add_patch(center_circle)
        
        # Draw penalty areas
        ax.plot([0, 16.5, 16.5, 0], [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
        ax.plot([field_length, field_length - 16.5, field_length - 16.5, field_length], 
                [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
        
        # Draw goal areas
        ax.plot([0, 5.5, 5.5, 0], [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
        ax.plot([field_length, field_length - 5.5, field_length - 5.5, field_length], 
                [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
        
        # Draw penalty spots
        ax.plot(11, field_width/2, 'ko', markersize=3)
        ax.plot(field_length - 11, field_width/2, 'ko', markersize=3)
        
        return [field_outline, center_line, ball] + home_players + away_players + home_labels + away_labels
    
    def update(frame):
        """Update animation for the given frame."""
        # Get predicted ball position
        ball_x = predictions.iloc[frame]['ball_x'] / 10  # Scale to field dimensions
        ball_y = predictions.iloc[frame]['ball_y'] / 10  # Scale to field dimensions
        ball.set_data([ball_x], [ball_y])
        
        # Update home players
        for i, (x_col, y_col) in enumerate(zip(home_x_cols, home_y_cols)):
            if frame < len(home_df):
                x = home_df.iloc[frame][x_col]
                y = home_df.iloc[frame][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    home_players[i].set_data([x], [y])
                    home_labels[i].set_position((x + 1, y + 1))
                    home_labels[i].set_text(x_col.replace('home_', '').replace('_x', ''))
                else:
                    home_players[i].set_data([], [])
                    home_labels[i].set_text('')
            else:
                home_players[i].set_data([], [])
                home_labels[i].set_text('')
        
        # Update away players
        for i, (x_col, y_col) in enumerate(zip(away_x_cols, away_y_cols)):
            if frame < len(away_df):
                x = away_df.iloc[frame][x_col]
                y = away_df.iloc[frame][y_col]
                if not np.isnan(x) and not np.isnan(y):
                    away_players[i].set_data([x], [y])
                    away_labels[i].set_position((x + 1, y + 1))
                    away_labels[i].set_text(x_col.replace('away_', '').replace('_x', ''))
                else:
                    away_players[i].set_data([], [])
                    away_labels[i].set_text('')
            else:
                away_players[i].set_data([], [])
                away_labels[i].set_text('')
        
        # Update title with frame number
        ax.set_title(f'Ball Coordinate Predictions - Frame {frame}')
        
        return [field_outline, center_line, ball] + home_players + away_players + home_labels + away_labels
    
    def update_slider(val):
        """Update plot when slider value changes."""
        frame = int(slider.val)
        update(frame)
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update_slider)
    
    # Initialize animation
    init()
    update(start_frame)
    
    if save_animation:
        print("Saving animation...")
        # Create animation
        frames = range(start_frame, start_frame + min(num_frames, len(predictions) - start_frame))
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
        
        # Save animation
        os.makedirs('animations', exist_ok=True)
        ani.save('animations/ball_predictions.mp4', writer='ffmpeg', fps=10)
        print("Animation saved to 'animations/ball_predictions.mp4'")
    
    plt.show()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load predictions
    predictions = load_predictions(args.predictions_file)
    
    # Load match data
    _, home_df, away_df = load_data(args.match_num)
    
    # Create animation
    create_animation(predictions, home_df, away_df, args.start_frame, args.num_frames, args.save_animation)

if __name__ == "__main__":
    main()
