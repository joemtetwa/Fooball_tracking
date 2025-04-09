def visualize_movement_patterns(ball_df, home_df, away_df, passes, shots, dribbles, args):
    """
    Create visualizations for detected movement patterns.
    
    Parameters:
    -----------
    ball_df : DataFrame
        DataFrame with ball position and velocity
    home_df, away_df : DataFrame
        DataFrames with player positions
    passes, shots, dribbles : list
        Detected movement patterns
    args : Namespace
        Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create field dimensions
    field_length = 105  # meters
    field_width = 68    # meters
    
    # 1. Plot all movement types on the field
    plt.figure(figsize=(12, 8))
    
    # Draw the field
    draw_soccer_field(plt.gca(), field_length, field_width)
    
    # Plot passes
    for from_player, to_player, start_frame, end_frame in passes:
        if start_frame < len(ball_df) and end_frame < len(ball_df):
            start_x = ball_df.iloc[start_frame - ball_df.iloc[0]['frame']]['ball_x']
            start_y = ball_df.iloc[start_frame - ball_df.iloc[0]['frame']]['ball_y']
            end_x = ball_df.iloc[end_frame - ball_df.iloc[0]['frame']]['ball_x']
            end_y = ball_df.iloc[end_frame - ball_df.iloc[0]['frame']]['ball_y']
            
            # Determine team color
            color = 'blue' if 'home_' in from_player else 'red'
            
            # Draw pass arrow
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                     color=color, alpha=0.3, width=0.2, head_width=1.0)
    
    # Plot shots
    for player_id, frame, shot_type, on_target in shots:
        if frame < len(ball_df):
            frame_idx = frame - ball_df.iloc[0]['frame']
            if 0 <= frame_idx < len(ball_df):
                x = ball_df.iloc[frame_idx]['ball_x']
                y = ball_df.iloc[frame_idx]['ball_y']
                
                # Determine color by shot outcome
                color = 'green' if on_target else 'orange'
                
                # Draw shot marker
                plt.scatter(x, y, color=color, s=100, marker='*', edgecolors='black', zorder=3)
    
    # Plot dribbles
    for player_id, start_frame, end_frame, distance in dribbles:
        if start_frame < len(ball_df) and end_frame < len(ball_df):
            start_idx = start_frame - ball_df.iloc[0]['frame']
            end_idx = end_frame - ball_df.iloc[0]['frame']
            
            if 0 <= start_idx < len(ball_df) and 0 <= end_idx < len(ball_df):
                # Get ball positions during dribble
                dribble_positions_x = ball_df.iloc[start_idx:end_idx+1]['ball_x'].values
                dribble_positions_y = ball_df.iloc[start_idx:end_idx+1]['ball_y'].values
                
                # Determine team color
                color = 'blue' if 'home_' in player_id else 'red'
                
                # Draw dribbling path
                plt.plot(dribble_positions_x, dribble_positions_y, color=color, linewidth=2, alpha=0.7)
    
    # Add legend
    home_pass = plt.Line2D([], [], color='blue', alpha=0.3, linewidth=2, label='Home Pass')
    away_pass = plt.Line2D([], [], color='red', alpha=0.3, linewidth=2, label='Away Pass')
    home_dribble = plt.Line2D([], [], color='blue', linewidth=2, alpha=0.7, label='Home Dribble')
    away_dribble = plt.Line2D([], [], color='red', linewidth=2, alpha=0.7, label='Away Dribble')
    on_target = plt.Line2D([], [], color='green', marker='*', linestyle='None', 
                          markersize=10, markeredgecolor='black', label='Shot On Target')
    off_target = plt.Line2D([], [], color='orange', marker='*', linestyle='None', 
                           markersize=10, markeredgecolor='black', label='Shot Off Target')
    
    plt.legend(handles=[home_pass, away_pass, home_dribble, away_dribble, on_target, off_target],
              loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    plt.title(f'Movement Patterns Analysis (Frames {args.start_frame}-{args.start_frame+args.num_frames})')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.tight_layout()
    
    plt.savefig(os.path.join(args.output_dir, 'movement_patterns_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot ball speed distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(ball_df['ball_speed'], bins=30, kde=True)
    plt.axvline(x=args.ball_speed_threshold, color='green', linestyle='--', 
               label=f'Pass Threshold ({args.ball_speed_threshold} m/s)')
    plt.axvline(x=args.shot_speed_threshold, color='red', linestyle='--', 
               label=f'Shot Threshold ({args.shot_speed_threshold} m/s)')
    plt.axvline(x=args.dribble_max_speed, color='blue', linestyle='--', 
               label=f'Dribble Threshold ({args.dribble_max_speed} m/s)')
    
    plt.title('Ball Speed Distribution')
    plt.xlabel('Ball Speed (m/s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(args.output_dir, 'ball_speed_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot pass networks
    if passes:
        # Separate passes by team
        home_passes = [p for p in passes if p[0].startswith('home_') and p[1].startswith('home_')]
        away_passes = [p for p in passes if p[0].startswith('away_') and p[1].startswith('away_')]
        
        # Calculate pass probabilities
        home_pass_matrix = calculate_pass_probabilities(home_passes, 'home_')
        away_pass_matrix = calculate_pass_probabilities(away_passes, 'away_')
        
        # Calculate average player positions
        avg_positions = calculate_average_player_positions(home_df, away_df, 
                                                         args.start_frame, args.start_frame + args.num_frames)
        
        # Visualize pass networks
        if not home_pass_matrix.empty:
            visualize_pass_network(home_pass_matrix, avg_positions, 
                                  title=f"Home Team Pass Network (Match {args.match_num})",
                                  min_probability=0.05)
            plt.savefig(os.path.join(args.output_dir, 'home_pass_network.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        if not away_pass_matrix.empty:
            visualize_pass_network(away_pass_matrix, avg_positions, 
                                  title=f"Away Team Pass Network (Match {args.match_num})",
                                  min_probability=0.05)
            plt.savefig(os.path.join(args.output_dir, 'away_pass_network.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Plot shot distribution
    if shots:
        plt.figure(figsize=(10, 8))
        
        # Draw the field
        draw_soccer_field(plt.gca(), field_length, field_width)
        
        # Separate by shot type
        on_target_shots = [(p, f, t, o) for p, f, t, o in shots if o]
        off_target_shots = [(p, f, t, o) for p, f, t, o in shots if not o]
        
        # Plot shot positions
        for player_id, frame, shot_type, _ in on_target_shots:
            if frame < len(ball_df):
                frame_idx = frame - ball_df.iloc[0]['frame']
                if 0 <= frame_idx < len(ball_df):
                    x = ball_df.iloc[frame_idx]['ball_x']
                    y = ball_df.iloc[frame_idx]['ball_y']
                    plt.scatter(x, y, color='green', s=100, marker='*', edgecolors='black', zorder=3)
        
        for player_id, frame, shot_type, _ in off_target_shots:
            if frame < len(ball_df):
                frame_idx = frame - ball_df.iloc[0]['frame']
                if 0 <= frame_idx < len(ball_df):
                    x = ball_df.iloc[frame_idx]['ball_x']
                    y = ball_df.iloc[frame_idx]['ball_y']
                    plt.scatter(x, y, color='orange', s=100, marker='*', edgecolors='black', zorder=3)
        
        # Add legend
        on_target = plt.Line2D([], [], color='green', marker='*', linestyle='None', 
                              markersize=10, markeredgecolor='black', label='On Target')
        off_target = plt.Line2D([], [], color='orange', marker='*', linestyle='None', 
                               markersize=10, markeredgecolor='black', label='Off Target')
        
        plt.legend(handles=[on_target, off_target], loc='upper center')
        plt.title(f'Shot Distribution (Frames {args.start_frame}-{args.start_frame+args.num_frames})')
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.tight_layout()
        
        plt.savefig(os.path.join(args.output_dir, 'shot_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create pie chart for shot outcomes
        plt.figure(figsize=(8, 8))
        labels = ['On Target', 'Off Target']
        sizes = [len(on_target_shots), len(off_target_shots)]
        colors = ['green', 'orange']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Shot Outcomes')
        plt.tight_layout()
        
        plt.savefig(os.path.join(args.output_dir, 'shot_outcomes_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Plot top dribblers
    if dribbles:
        # Aggregate dribbling distance by player
        player_dribbling = defaultdict(float)
        for player_id, _, _, distance in dribbles:
            player_dribbling[player_id] += distance
        
        # Sort players by dribbling distance
        top_dribblers = sorted(player_dribbling.items(), key=lambda x: x[1], reverse=True)[:10]
        
        plt.figure(figsize=(12, 6))
        players = [p[0].replace('home_', 'H_').replace('away_', 'A_') for p in top_dribblers]
        distances = [p[1] for p in top_dribblers]
        colors = ['blue' if 'home_' in p[0] else 'red' for p in top_dribblers]
        
        plt.bar(players, distances, color=colors)
        plt.xlabel('Player ID')
        plt.ylabel('Total Dribbling Distance (meters)')
        plt.title('Top 10 Dribblers by Distance')
        plt.xticks(rotation=45, ha='right')
        
        # Add team labels in legend
        home_patch = plt.Rectangle((0, 0), 1, 1, color='blue', label='Home Team')
        away_patch = plt.Rectangle((0, 0), 1, 1, color='red', label='Away Team')
        plt.legend(handles=[home_patch, away_patch])
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'top_dribblers.png'), dpi=300, bbox_inches='tight')
        plt.close()

def draw_soccer_field(ax, length=105, width=68):
    """Draw a soccer field."""
    # Field outline
    ax.plot([0, 0, length, length, 0], [0, width, width, 0, 0], 'k-')
    
    # Halfway line
    ax.plot([length/2, length/2], [0, width], 'k-')
    
    # Center circle
    center_circle = plt.Circle((length/2, width/2), 9.15, fill=False, color='k')
    ax.add_patch(center_circle)
    
    # Penalty areas
    # Home penalty area
    ax.plot([0, 16.5, 16.5, 0], [width/2 - 20.16, width/2 - 20.16, width/2 + 20.16, width/2 + 20.16], 'k-')
    # Away penalty area
    ax.plot([length, length - 16.5, length - 16.5, length], 
           [width/2 - 20.16, width/2 - 20.16, width/2 + 20.16, width/2 + 20.16], 'k-')
    
    # Goal areas
    # Home goal area
    ax.plot([0, 5.5, 5.5, 0], [width/2 - 9.16, width/2 - 9.16, width/2 + 9.16, width/2 + 9.16], 'k-')
    # Away goal area
    ax.plot([length, length - 5.5, length - 5.5, length], 
           [width/2 - 9.16, width/2 - 9.16, width/2 + 9.16, width/2 + 9.16], 'k-')
    
    # Goals
    # Home goal
    ax.plot([0, -2, -2, 0], [width/2 - 3.66, width/2 - 3.66, width/2 + 3.66, width/2 + 3.66], 'k-')
    # Away goal
    ax.plot([length, length + 2, length + 2, length], 
           [width/2 - 3.66, width/2 - 3.66, width/2 + 3.66, width/2 + 3.66], 'k-')
    
    # Set axis limits to show a bit of space around the field
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set aspect ratio to ensure the field looks right
    ax.set_aspect('equal')

def analyze_movement_patterns(predictions, home_df, away_df, args):
    """
    Analyze movement patterns from enhanced ball predictions.
    
    Parameters:
    -----------
    predictions : DataFrame
        DataFrame containing predicted ball coordinates
    home_df, away_df : DataFrame
        DataFrames with player positions
    args : Namespace
        Command-line arguments
    """
    print("Analyzing movement patterns...")
    
    # Create enhanced ball DataFrame
    ball_df = create_enhanced_ball_df(predictions, args.start_frame, args.num_frames)
    
    # Detect passes
    print("Detecting passes...")
    passes = detect_passes(
        ball_df, home_df, away_df, 
        possession_radius=args.possession_radius,
        ball_speed_threshold=args.ball_speed_threshold,
        prediction_mode=True  # Modified implementation of detect_passes
    )
    
    # Detect shots
    print("Detecting shots...")
    shots = detect_shots(
        ball_df, home_df, away_df,
        shot_speed_threshold=args.shot_speed_threshold,
        possession_radius=args.possession_radius
    )
    
    # Detect dribbles
    print("Detecting dribbles...")
    dribbles = detect_dribbles(
        ball_df, home_df, away_df,
        possession_radius=args.possession_radius,
        dribble_max_speed=args.dribble_max_speed
    )
    
    # Print summary statistics
    print("\nMovement Pattern Analysis Summary:")
    print(f"Total Passes: {len(passes)}")
    home_passes = len([p for p in passes if p[0].startswith('home_') and p[1].startswith('home_')])
    away_passes = len([p for p in passes if p[0].startswith('away_') and p[1].startswith('away_')])
    print(f"  Home Team Passes: {home_passes}")
    print(f"  Away Team Passes: {away_passes}")
    
    print(f"Total Shots: {len(shots)}")
    on_target = len([s for s in shots if s[3]])
    off_target = len([s for s in shots if not s[3]])
    print(f"  On Target: {on_target}")
    print(f"  Off Target: {off_target}")
    print(f"  Shot Accuracy: {on_target/len(shots)*100:.1f}%" if shots else "  Shot Accuracy: N/A")
    
    print(f"Total Dribbles: {len(dribbles)}")
    home_dribbles = len([d for d in dribbles if d[0].startswith('home_')])
    away_dribbles = len([d for d in dribbles if d[0].startswith('away_')])
    print(f"  Home Team Dribbles: {home_dribbles}")
    print(f"  Away Team Dribbles: {away_dribbles}")
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_movement_patterns(ball_df, home_df, away_df, passes, shots, dribbles, args)
    
    # Save data to CSV files
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save ball data
    ball_df.to_csv(os.path.join(args.output_dir, 'enhanced_ball_data.csv'), index=False)
    
    # Save passes
    passes_df = pd.DataFrame(passes, columns=['from_player', 'to_player', 'start_frame', 'end_frame'])
    passes_df.to_csv(os.path.join(args.output_dir, 'detected_passes.csv'), index=False)
    
    # Save shots
    shots_df = pd.DataFrame(shots, columns=['player_id', 'frame', 'shot_type', 'on_target'])
    shots_df.to_csv(os.path.join(args.output_dir, 'detected_shots.csv'), index=False)
    
    # Save dribbles
    dribbles_df = pd.DataFrame(dribbles, columns=['player_id', 'start_frame', 'end_frame', 'distance'])
    dribbles_df.to_csv(os.path.join(args.output_dir, 'detected_dribbles.csv'), index=False)
    
    return ball_df, passes, shots, dribbles

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load match data
    print(f"Loading match {args.match_num} data...")
    _, home_df, away_df = load_data(args.match_num, prediction_mode=True)
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_file}...")
    predictions = pd.read_csv(args.predictions_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Analyze movement patterns
        ball_df, passes, shots, dribbles = analyze_movement_patterns(predictions, home_df, away_df, args)
        print("Analysis complete. Results saved to:", args.output_dir)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
