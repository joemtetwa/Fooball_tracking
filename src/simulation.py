import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, Circle
import os

def load_match_data(match_number):
    """Load and combine home and away team data for a match."""
    data_dir = os.path.join('Data', f'match_{match_number}')
    
    # Load home and away data
    home_data = pd.read_csv(os.path.join(data_dir, 'Home.csv'))
    away_data = pd.read_csv(os.path.join(data_dir, 'Away.csv'))
    
    # Ensure both dataframes have the same length
    min_len = min(len(home_data), len(away_data))
    home_data = home_data.iloc[:min_len]
    away_data = away_data.iloc[:min_len]
    
    # Combine data
    match_data = pd.concat([home_data, away_data], axis=1)
    
    # Remove duplicate columns
    match_data = match_data.loc[:,~match_data.columns.duplicated()]
    
    return match_data

class MatchSimulation:
    def __init__(self, match_number=1, field_width=1000, field_height=2000):
        # Load match data
        self.data = load_match_data(match_number)
        self.current_frame = 0
        self.field_width = field_width
        self.field_height = field_height
        
        # Setup the figure and animation
        self.fig, self.ax = plt.subplots(figsize=(10, 15))
        self.fig.patch.set_facecolor('#1a1a1a')  # Dark background
        
        # Initialize storage for artists
        self.players = {}
        self.ball = None
        self.possession_text = None
        self.time_text = None
        self.trail_dots = []  # Store ball trail
        
        # Setup the soccer field
        self._setup_field()
        
    def _setup_field(self):
        """Setup the soccer field appearance."""
        self.ax.set_facecolor('#238823')  # Green field
        
        # Field outline
        self.ax.add_patch(Rectangle((0, 0), self.field_width, self.field_height,
                                  fill=False, color='white', linewidth=2))
        
        # Centerline
        plt.axhline(y=self.field_height/2, color='white', linewidth=2)
        
        # Center circle
        center_circle = Circle((self.field_width/2, self.field_height/2),
                             radius=91.5, fill=False, color='white')
        self.ax.add_patch(center_circle)
        
        # Penalty areas
        penalty_width = 440
        penalty_height = 180
        penalty_x = (self.field_width - penalty_width) / 2
        
        # Top penalty area
        self.ax.add_patch(Rectangle((penalty_x, self.field_height - penalty_height),
                                  penalty_width, penalty_height,
                                  fill=False, color='white'))
        
        # Bottom penalty area
        self.ax.add_patch(Rectangle((penalty_x, 0),
                                  penalty_width, penalty_height,
                                  fill=False, color='white'))
        
        # Set axis limits
        self.ax.set_xlim(-50, self.field_width + 50)
        self.ax.set_ylim(-50, self.field_height + 50)
        
        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
    def init_animation(self):
        """Initialize the animation."""
        # Create ball
        self.ball = Circle((0, 0), radius=15, color='white', zorder=5)
        self.ax.add_patch(self.ball)
        
        # Initialize ball trail
        for _ in range(10):  # 10 trail dots
            dot = Circle((0, 0), radius=5, color='white', alpha=0.2, zorder=4)
            self.trail_dots.append(dot)
            self.ax.add_patch(dot)
        
        # Create players for both teams
        home_players = [col.split('_')[1] for col in self.data.columns if 'home_' in col and '_x' in col]
        away_players = [col.split('_')[1] for col in self.data.columns if 'away_' in col and '_x' in col]
        
        # Home team (blue)
        for player_id in home_players:
            circle = Circle((0, 0), radius=20, color='blue', alpha=0.7, zorder=3)
            self.players[f'home_{player_id}'] = circle
            self.ax.add_patch(circle)
        
        # Away team (red)
        for player_id in away_players:
            circle = Circle((0, 0), radius=20, color='red', alpha=0.7, zorder=3)
            self.players[f'away_{player_id}'] = circle
            self.ax.add_patch(circle)
        
        # Add possession and time text
        self.possession_text = self.ax.text(50, self.field_height + 20,
                                          '', color='white', fontsize=12)
        self.time_text = self.ax.text(self.field_width - 150, self.field_height + 20,
                                    '', color='white', fontsize=12)
        
        return [self.ball] + self.trail_dots + list(self.players.values()) + [self.possession_text, self.time_text]
    
    def update(self, frame):
        """Update the animation for each frame."""
        self.current_frame = frame
        
        # Update ball position and trail
        ball_x = self.data.iloc[frame]['ball_x']
        ball_y = self.data.iloc[frame]['ball_y']
        self.ball.center = (ball_x, ball_y)
        
        # Update trail dots
        for i, dot in enumerate(self.trail_dots):
            if frame - i*2 >= 0:  # Only show trail for existing frames
                old_x = self.data.iloc[frame - i*2]['ball_x']
                old_y = self.data.iloc[frame - i*2]['ball_y']
                dot.center = (old_x, old_y)
                dot.set_alpha(0.2 - i*0.02)  # Fade out older dots
        
        # Update player positions
        for player_id, circle in self.players.items():
            x = self.data.iloc[frame][f'{player_id}_x']
            y = self.data.iloc[frame][f'{player_id}_y']
            circle.center = (x, y)
        
        # Update time text
        time = self.data.iloc[frame]['Time']
        period = self.data.iloc[frame]['IdPeriod']
        self.time_text.set_text(f'Period: {period} | Time: {time:.2f}')
        
        # Determine possession based on closest player to ball
        min_dist = float('inf')
        possessing_team = 'unknown'
        
        for player_id in self.players.keys():
            player_x = self.data.iloc[frame][f'{player_id}_x']
            player_y = self.data.iloc[frame][f'{player_id}_y']
            dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                possessing_team = 'Home' if 'home_' in player_id else 'Away'
        
        self.possession_text.set_text(f'Possession: {possessing_team}')
        
        return [self.ball] + self.trail_dots + list(self.players.values()) + [self.possession_text, self.time_text]
    
    def animate(self, interval=50):
        """Create and display the animation."""
        anim = animation.FuncAnimation(self.fig, self.update,
                                     init_func=self.init_animation,
                                     frames=len(self.data),
                                     interval=interval,  # milliseconds between frames
                                     blit=True)
        plt.show()
        return anim

if __name__ == "__main__":
    # Create simulation from match 1 data
    sim = MatchSimulation(match_number=1)
    sim.animate(interval=50)  # 50ms between frames = 20 FPS
