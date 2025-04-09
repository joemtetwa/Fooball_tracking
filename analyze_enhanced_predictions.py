import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import distance
from scipy.stats import pearsonr

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze Enhanced Ball Coordinate Predictions')
    
    parser.add_argument('--original_file', type=str, default='predictions/match_4_ball_predictions.csv',
                        help='Path to the original predictions CSV file')
    parser.add_argument('--enhanced_file', type=str, default='predictions/match_4_enhanced_predictions.csv',
                        help='Path to the enhanced predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_enhanced',
                        help='Directory to save analysis results')
    
    return parser.parse_args()

def calculate_metrics(original, enhanced):
    """
    Calculate metrics comparing original and enhanced predictions.
    
    Args:
        original: DataFrame with original predictions
        enhanced: DataFrame with enhanced predictions
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Extract coordinates
    orig_x = original['ball_x'].values
    orig_y = original['ball_y'].values
    enh_x = enhanced['ball_x'].values
    enh_y = enhanced['ball_y'].values
    
    # Calculate distances between original and enhanced predictions
    distances = np.sqrt((orig_x - enh_x)**2 + (orig_y - enh_y)**2)
    
    # Calculate velocities
    orig_vx = np.diff(orig_x)
    orig_vy = np.diff(orig_y)
    orig_speed = np.sqrt(orig_vx**2 + orig_vy**2)
    
    enh_vx = np.diff(enh_x)
    enh_vy = np.diff(enh_y)
    enh_speed = np.sqrt(enh_vx**2 + enh_vy**2)
    
    # Calculate accelerations
    orig_ax = np.diff(orig_vx)
    orig_ay = np.diff(orig_vy)
    orig_acc = np.sqrt(orig_ax**2 + orig_ay**2)
    
    enh_ax = np.diff(enh_vx)
    enh_ay = np.diff(enh_vy)
    enh_acc = np.sqrt(enh_ax**2 + enh_ay**2)
    
    # Calculate smoothness (inverse of average acceleration)
    orig_smoothness = 1.0 / (np.mean(orig_acc) + 1e-10)
    enh_smoothness = 1.0 / (np.mean(enh_acc) + 1e-10)
    
    # Calculate correlation between original and enhanced trajectories
    x_corr, _ = pearsonr(orig_x, enh_x)
    y_corr, _ = pearsonr(orig_y, enh_y)
    
    # Calculate metrics
    metrics = {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances),
        
        'orig_mean_speed': np.mean(orig_speed),
        'orig_max_speed': np.max(orig_speed),
        'orig_std_speed': np.std(orig_speed),
        
        'enh_mean_speed': np.mean(enh_speed),
        'enh_max_speed': np.max(enh_speed),
        'enh_std_speed': np.std(enh_speed),
        
        'orig_mean_acc': np.mean(orig_acc),
        'orig_max_acc': np.max(orig_acc),
        'orig_std_acc': np.std(orig_acc),
        
        'enh_mean_acc': np.mean(enh_acc),
        'enh_max_acc': np.max(enh_acc),
        'enh_std_acc': np.std(enh_acc),
        
        'orig_smoothness': orig_smoothness,
        'enh_smoothness': enh_smoothness,
        'smoothness_improvement': enh_smoothness / orig_smoothness,
        
        'x_correlation': x_corr,
        'y_correlation': y_corr
    }
    
    return metrics

def analyze_predictions(original, enhanced, output_dir):
    """
    Analyze original and enhanced ball coordinate predictions.
    
    Args:
        original: DataFrame with original predictions
        enhanced: DataFrame with enhanced predictions
        output_dir: Directory to save analysis results
    """
    print("Analyzing predictions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = calculate_metrics(original, enhanced)
    
    # Print metrics
    print("\nComparison Metrics:")
    print(f"Mean Distance between Original and Enhanced: {metrics['mean_distance']:.2f} units")
    print(f"Maximum Distance: {metrics['max_distance']:.2f} units")
    print(f"X Correlation: {metrics['x_correlation']:.4f}")
    print(f"Y Correlation: {metrics['y_correlation']:.4f}")
    
    print("\nSpeed Metrics:")
    print(f"Original Mean Speed: {metrics['orig_mean_speed']:.2f} units/frame")
    print(f"Enhanced Mean Speed: {metrics['enh_mean_speed']:.2f} units/frame")
    print(f"Original Max Speed: {metrics['orig_max_speed']:.2f} units/frame")
    print(f"Enhanced Max Speed: {metrics['enh_max_speed']:.2f} units/frame")
    
    print("\nSmoothness Metrics:")
    print(f"Original Smoothness: {metrics['orig_smoothness']:.4f}")
    print(f"Enhanced Smoothness: {metrics['enh_smoothness']:.4f}")
    print(f"Smoothness Improvement: {metrics['smoothness_improvement']:.2f}x")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)
    
    # Extract coordinates
    orig_x = original['ball_x'].values
    orig_y = original['ball_y'].values
    enh_x = enhanced['ball_x'].values
    enh_y = enhanced['ball_y'].values
    
    # Calculate distances between original and enhanced predictions
    distances = np.sqrt((orig_x - enh_x)**2 + (orig_y - enh_y)**2)
    
    # Plot distance between original and enhanced predictions
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('Distance Between Original and Enhanced Predictions')
    plt.xlabel('Frame')
    plt.ylabel('Distance (units)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_distances.png'))
    
    # Plot speed comparison
    orig_vx = np.diff(orig_x)
    orig_vy = np.diff(orig_y)
    orig_speed = np.sqrt(orig_vx**2 + orig_vy**2)
    
    enh_vx = np.diff(enh_x)
    enh_vy = np.diff(enh_y)
    enh_speed = np.sqrt(enh_vx**2 + enh_vy**2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(orig_speed, 'y-', alpha=0.7, label='Original')
    plt.plot(enh_speed, 'g-', alpha=0.7, label='Enhanced')
    plt.title('Ball Speed Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Speed (units/frame)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'))
    
    # Plot acceleration comparison
    orig_ax = np.diff(orig_vx)
    orig_ay = np.diff(orig_vy)
    orig_acc = np.sqrt(orig_ax**2 + orig_ay**2)
    
    enh_ax = np.diff(enh_vx)
    enh_ay = np.diff(enh_vy)
    enh_acc = np.sqrt(enh_ax**2 + enh_ay**2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(orig_acc, 'y-', alpha=0.7, label='Original')
    plt.plot(enh_acc, 'g-', alpha=0.7, label='Enhanced')
    plt.title('Ball Acceleration Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Acceleration (units/frame²)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'acceleration_comparison.png'))
    
    # Create heatmap comparison
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.hist2d(orig_x / 10, orig_y / 10, bins=50, cmap='hot')
    plt.colorbar(label='Frequency')
    plt.title('Original Ball Position Heatmap')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    
    plt.subplot(1, 2, 2)
    plt.hist2d(enh_x / 10, enh_y / 10, bins=50, cmap='hot')
    plt.colorbar(label='Frequency')
    plt.title('Enhanced Ball Position Heatmap')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_comparison.png'))
    
    # Create direction change analysis
    # Calculate direction changes (angle between consecutive velocity vectors)
    orig_angles = []
    enh_angles = []
    
    for i in range(len(orig_vx) - 1):
        # Original direction change
        v1 = np.array([orig_vx[i], orig_vy[i]])
        v2 = np.array([orig_vx[i+1], orig_vy[i+1]])
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # Clip to handle floating point errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            orig_angles.append(angle)
        else:
            orig_angles.append(0)
        
        # Enhanced direction change
        v1 = np.array([enh_vx[i], enh_vy[i]])
        v2 = np.array([enh_vx[i+1], enh_vy[i+1]])
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # Clip to handle floating point errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            enh_angles.append(angle)
        else:
            enh_angles.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(orig_angles, 'y-', alpha=0.7, label='Original')
    plt.plot(enh_angles, 'g-', alpha=0.7, label='Enhanced')
    plt.title('Direction Change Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Direction Change (degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'direction_change_comparison.png'))
    
    # Create histogram of direction changes
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(orig_angles, bins=36, range=(0, 180), alpha=0.7, color='y')
    plt.title('Original Direction Changes')
    plt.xlabel('Direction Change (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(enh_angles, bins=36, range=(0, 180), alpha=0.7, color='g')
    plt.title('Enhanced Direction Changes')
    plt.xlabel('Direction Change (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_change_histogram.png'))
    
    print(f"Analysis results saved to {output_dir}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load predictions
    print(f"Loading original predictions from {args.original_file}...")
    original = pd.read_csv(args.original_file)
    
    print(f"Loading enhanced predictions from {args.enhanced_file}...")
    enhanced = pd.read_csv(args.enhanced_file)
    
    # Analyze predictions
    analyze_predictions(original, enhanced, args.output_dir)

if __name__ == "__main__":
    main()
