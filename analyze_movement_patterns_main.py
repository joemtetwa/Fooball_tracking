import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import functionality from the two script parts
from analyze_movement_patterns import (
    parse_args, calculate_ball_velocity, create_enhanced_ball_df,
    detect_shots, detect_dribbles
)
from analyze_movement_patterns_part2 import (
    visualize_movement_patterns, draw_soccer_field, analyze_movement_patterns, main
)

if __name__ == "__main__":
    # Execute the main function from part2
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
