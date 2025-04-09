import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Ball Coordinate Predictions')
    
    parser.add_argument('--predictions_file', type=str, default='predictions/match_4_ball_predictions.csv',
                        help='Path to the predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    
    return parser.parse_args()

def analyze_predictions(predictions, output_dir):
    """Analyze predicted ball coordinates."""
    print("Analyzing predictions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract ball coordinates
    ball_x = predictions['ball_x'].values
    ball_y = predictions['ball_y'].values
    
    # Calculate statistics
    x_mean = np.mean(ball_x)
    y_mean = np.mean(ball_y)
    x_std = np.std(ball_x)
    y_std = np.std(ball_y)
    
    # Calculate velocities
    x_vel = np.diff(ball_x)
    y_vel = np.diff(ball_y)
    speed = np.sqrt(x_vel**2 + y_vel**2)
    
    # Calculate acceleration
    x_acc = np.diff(x_vel)
    y_acc = np.diff(y_vel)
    acceleration = np.sqrt(x_acc**2 + y_acc**2)
    
    # Print statistics
    print(f"Ball X Position: Mean = {x_mean:.2f}, Std = {x_std:.2f}")
    print(f"Ball Y Position: Mean = {y_mean:.2f}, Std = {y_std:.2f}")
    print(f"Ball Speed: Mean = {np.mean(speed):.2f}, Max = {np.max(speed):.2f}")
    print(f"Ball Acceleration: Mean = {np.mean(acceleration):.2f}, Max = {np.max(acceleration):.2f}")
    
    # Create heatmap of ball positions
    plt.figure(figsize=(10, 7))
    plt.hist2d(ball_x / 10, ball_y / 10, bins=50, cmap='hot')
    plt.colorbar(label='Frequency')
    plt.title('Ball Position Heatmap')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.savefig(os.path.join(output_dir, 'ball_position_heatmap.png'))
    
    # Plot ball position over time
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ball_x)
    plt.title('Ball X Position Over Time')
    plt.xlabel('Frame')
    plt.ylabel('X Position')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(ball_y)
    plt.title('Ball Y Position Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ball_position_over_time.png'))
    
    # Plot ball speed over time
    plt.figure(figsize=(10, 6))
    plt.plot(speed)
    plt.title('Ball Speed Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Speed')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ball_speed_over_time.png'))
    
    # Plot ball trajectory
    plt.figure(figsize=(10, 7))
    
    # Set field dimensions
    field_length = 105  # meters
    field_width = 68    # meters
    
    # Draw field
    plt.plot([0, 0, field_length, field_length, 0], [0, field_width, field_width, 0, 0], 'k-')
    plt.plot([field_length/2, field_length/2], [0, field_width], 'k-')
    
    # Draw center circle
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15, fill=False, color='k')
    plt.gca().add_patch(center_circle)
    
    # Draw penalty areas
    plt.plot([0, 16.5, 16.5, 0], [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
    plt.plot([field_length, field_length - 16.5, field_length - 16.5, field_length], 
             [field_width/2 - 20.16, field_width/2 - 20.16, field_width/2 + 20.16, field_width/2 + 20.16], 'k-')
    
    # Draw goal areas
    plt.plot([0, 5.5, 5.5, 0], [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
    plt.plot([field_length, field_length - 5.5, field_length - 5.5, field_length], 
             [field_width/2 - 9.16, field_width/2 - 9.16, field_width/2 + 9.16, field_width/2 + 9.16], 'k-')
    
    # Draw penalty spots
    plt.plot(11, field_width/2, 'ko', markersize=3)
    plt.plot(field_length - 11, field_width/2, 'ko', markersize=3)
    
    # Plot ball trajectory (sample every 10 frames for clarity)
    sample_rate = 10
    plt.plot(ball_x[::sample_rate] / 10, ball_y[::sample_rate] / 10, 'r-', alpha=0.5, linewidth=1)
    plt.scatter(ball_x[::sample_rate] / 10, ball_y[::sample_rate] / 10, c=range(len(ball_x[::sample_rate])), 
                cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label='Frame (sampled)')
    
    plt.title('Ball Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ball_trajectory.png'))
    
    print(f"Analysis results saved to {output_dir}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load predictions
    predictions = pd.read_csv(args.predictions_file)
    
    # Analyze predictions
    analyze_predictions(predictions, args.output_dir)

if __name__ == "__main__":
    main()
