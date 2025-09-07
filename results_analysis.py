# Import the notebook-friendly functions
from nba_results_notebook import (
    show_game_summary, 
    show_run_timeline, 
    show_performance_stats,
    quick_summary,
    load_results_df,
    get_successful_runs,
    get_run_details
)

# For additional analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Show the beautiful game summary table
summary_df = show_game_summary()
summary_df