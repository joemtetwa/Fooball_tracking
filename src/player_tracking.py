import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, Circle
import os

def scale_coordinates(data, field_width=2000, field_height=1000):
    """Scale all player and ball coordinates to match field dimensions."""
    scaled_data = data.copy()
    
    # Get x and y columns
    x_cols = [col for col in data.columns if col.endswith('_x')]
    y_cols = [col for col in data.columns if col.endswith('_y')]
    
    print(f"\nFound {len(x_cols)} x-coordinate columns and {len(y_cols)} y-coordinate columns")
    
    # Get global coordinate ranges - these represent the actual field dimensions
    x_min = -5490  # Left boundary
    x_max = 5490   # Right boundary (length of the pitch)
    y_min = -3720  # Bottom boundary
    y_max = 3720   # Top boundary (width of the pitch)
    
    print(f"\nUsing fixed coordinate ranges:")
    print(f"X: {x_min:.2f} to {x_max:.2f}")
    print(f"Y: {y_min:.2f} to {y_max:.2f}")
    
    # Scale all x and y coordinates
    for x_col in x_cols:
        entity = x_col.replace('_x', '')
        scaled_col = f'{entity}_x_scaled'
        # Scale x-coordinates to field width (represents length of the pitch)
        scaled_data[scaled_col] = (data[x_col] - x_min) * field_width / (x_max - x_min)
        print(f"Created scaled column: {scaled_col}")
        
    for y_col in y_cols:
        entity = y_col.replace('_y', '')
        scaled_col = f'{entity}_y_scaled'
        # Scale y-coordinates to field height (represents width of the pitch)
        # Flip y-coordinates by subtracting from field_height since y increases downward in matplotlib
        scaled_data[scaled_col] = field_height - (data[y_col] - y_min) * field_height / (y_max - y_min)
        print(f"Created scaled column: {scaled_col}")
    
    # Calculate velocities for players (excluding ball)
    player_x_cols = [col for col in x_cols if 'ball' not in col]
    for x_col in player_x_cols:
        entity = x_col.replace('_x', '')
        scaled_data[f'{entity}_vx'] = scaled_data[f'{entity}_x_scaled'].diff() / 0.1  # x velocity is normal
        scaled_data[f'{entity}_vy'] = -scaled_data[f'{entity}_y_scaled'].diff() / 0.1  # y velocity is flipped
        scaled_data[f'{entity}_speed'] = np.sqrt(
            scaled_data[f'{entity}_vx']**2 + scaled_data[f'{entity}_vy']**2
        ).fillna(0)
    
    # Print first few rows of scaled coordinates for verification
    print("\nFirst few rows of scaled coordinates:")
    scaled_cols = [col for col in scaled_data.columns if 'scaled' in col][:4]  # First 4 scaled columns
    print(scaled_data[scaled_cols].head())
    
    return scaled_data

class MatchSimulation:
    def __init__(self, match_number=1, field_width=2000, field_height=1000):
        # Set field dimensions first
        self.field_width = field_width
        self.field_height = field_height
        self.trail_length = 20
        
        # Initialize storage for artists
        self.player_dots = {}
        self.player_trails = {}
        self.ball_dot = None
        self.ball_trail = []
        self.time_text = None
        
        # Load match data
        print(f"\nLoading match {match_number} data...")
        self.data = self.load_match_data(match_number)
        
        # Get player IDs (excluding ball)
        self.player_ids = [col[:-2] for col in self.data.columns 
                          if col.endswith('_x') and 'ball' not in col]
        print(f"\nTracking {len(self.player_ids)} players and ball")
        print(f"Player IDs: {self.player_ids}")
        
        # Initialize player trails after we have player IDs
        self.player_trails = {player_id: [] for player_id in self.player_ids}
        
        # Verify column list
        required_columns = ['Time', 'IdPeriod'] + [f'{player_id}_x_scaled' for player_id in self.player_ids] + [f'{player_id}_y_scaled' for player_id in self.player_ids] + ['ball_x_scaled', 'ball_y_scaled']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")
        
        # Setup the figure
        self.fig = plt.figure(figsize=(15, 8))  # Adjusted aspect ratio
        self.field_ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#1a1a1a')
        
        # Setup the soccer field
        self._setup_field()
        
    def load_match_data(self, match_number):
        """Load match data from CSV files."""
        # Load home team data
        home_file = os.path.join('Data', f'match_{match_number}', 'Home.csv')
        home_data = pd.read_csv(home_file)
        
        # Load away team data
        away_file = os.path.join('Data', f'match_{match_number}', 'Away.csv')
        away_data = pd.read_csv(away_file)
        
        # Rename columns to avoid duplicates
        home_columns = home_data.columns
        away_columns = away_data.columns
        
        # Rename ball columns in away data to avoid duplicates
        away_data = away_data.rename(columns={
            'ball_x': 'ball_x_away',
            'ball_y': 'ball_y_away'
        })
        
        # Combine home and away data
        all_data = pd.concat([home_data, away_data], axis=1)
        
        # Use ball data from home team (they should be the same)
        all_data['ball_x'] = home_data['ball_x']
        all_data['ball_y'] = home_data['ball_y']
        
        # Scale coordinates
        scaled_data = scale_coordinates(all_data, self.field_width, self.field_height)
        
        return scaled_data

    def _setup_field(self):
        """Setup the soccer field appearance."""
        self.field_ax.set_facecolor('#238823')  # Green field
        
        # Field outline
        self.field_ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height,
                                        fill=False, color='white', linewidth=2))
        
        # Centerline
        self.field_ax.axvline(x=self.field_width/2, color='white', linewidth=2)
        
        # Center circle
        center_circle = Circle((self.field_width/2, self.field_height/2),
                             radius=91.5, fill=False, color='white')
        self.field_ax.add_patch(center_circle)
        
        # Add penalty boxes
        penalty_width = 320  # Approx 16.5m in field units
        penalty_height = 440  # Approx 40.3m in field units
        
        # Left penalty box
        self.field_ax.add_patch(Rectangle(
            (0, self.field_height/2 - penalty_height/2),
            penalty_width, penalty_height,
            fill=False, color='white', linewidth=2
        ))
        
        # Right penalty box
        self.field_ax.add_patch(Rectangle(
            (self.field_width - penalty_width, self.field_height/2 - penalty_height/2),
            penalty_width, penalty_height,
            fill=False, color='white', linewidth=2
        ))
        
        # Set axis limits with some padding
        self.field_ax.set_xlim(-50, self.field_width + 50)
        self.field_ax.set_ylim(-50, self.field_height + 50)
        
        # Remove axis ticks
        self.field_ax.set_xticks([])
        self.field_ax.set_yticks([])
        
    def init(self):
        """Initialize the animation."""
        # Create initial ball dot
        self.ball_dot = self.field_ax.plot([], [], 'ro', markersize=8)[0]
        
        # Create initial player dots
        for player_id in self.player_ids:
            color = 'yellow' if 'home' in player_id else 'blue'
            self.player_dots[player_id] = self.field_ax.plot([], [], 'o', 
                                                           color=color, 
                                                           markersize=10)[0]
        
        # Create time text
        self.time_text = self.field_ax.text(0.01, 0.95, '',
                                          transform=self.field_ax.transAxes,
                                          color='white', fontsize=12)
        
        # Return all artists that need to be updated
        return [self.ball_dot, self.time_text] + list(self.player_dots.values())

    def animate(self, interval=50):
        """Create and display the animation."""
        print("\nStarting match simulation...")
        print(f"Field dimensions: {self.field_width}x{self.field_height}")
        print(f"Number of frames: {len(self.data)}")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update,
            init_func=self.init,
            frames=len(self.data),
            interval=interval,
            blit=True
        )
        
        plt.show()
        return anim

    def update(self, frame):
        """Update the animation for the given frame."""
        # Update ball position
        ball_x = self.data.iloc[frame]['ball_x_scaled']
        ball_y = self.data.iloc[frame]['ball_y_scaled']
        self.ball_dot.set_data([ball_x], [ball_y])
        
        # Update ball trail
        if frame > 0:
            trail_start = max(0, frame - self.trail_length)
            ball_trail_x = self.data.iloc[trail_start:frame+1]['ball_x_scaled']
            ball_trail_y = self.data.iloc[trail_start:frame+1]['ball_y_scaled']
            ball_trail_line = self.field_ax.plot(ball_trail_x, ball_trail_y, 'r-', alpha=0.3)[0]
            self.ball_trail.append(ball_trail_line)

        # Update player positions
        for player_id in self.player_ids:
            x = self.data.iloc[frame][f'{player_id}_x_scaled']
            y = self.data.iloc[frame][f'{player_id}_y_scaled']
            self.player_dots[player_id].set_data([x], [y])
            
            # Update player trail
            if frame > 0:
                trail_start = max(0, frame - self.trail_length)
                trail_x = self.data.iloc[trail_start:frame+1][f'{player_id}_x_scaled']
                trail_y = self.data.iloc[trail_start:frame+1][f'{player_id}_y_scaled']
                color = 'yellow' if 'home' in player_id else 'blue'
                trail_line = self.field_ax.plot(trail_x, trail_y, '-', 
                                              color=color, alpha=0.3)[0]
                self.player_trails[player_id].append(trail_line)

        # Update time text
        period = 1  # You can update this based on your data
        time_seconds = frame * 0.1  # Assuming 10 fps
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        self.time_text.set_text(f'Period: {period} | Time: {minutes:02d}:{seconds:02d}')
        
        # Clear old trails periodically to prevent memory issues
        if frame % 50 == 0:
            for trail in self.ball_trail[:-self.trail_length]:
                if trail:
                    trail.remove()
            self.ball_trail = self.ball_trail[-self.trail_length:]
            
            for player_trails in self.player_trails.values():
                for trail in player_trails[:-self.trail_length]:
                    if trail:
                        trail.remove()
                player_trails[:] = player_trails[-self.trail_length:]
        
        # Return all artists that need to be updated
        return [self.ball_dot, self.time_text] + \
               list(self.player_dots.values()) + \
               [line for trail in self.player_trails.values() for line in trail] + \
               self.ball_trail

if __name__ == "__main__":
    # Create match simulation with all players and ball
    sim = MatchSimulation(match_number=1)
    sim.animate(interval=50)  # 50ms between frames = 20 FPS
