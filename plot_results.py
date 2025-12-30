import sys
sys.path.insert(0, '/home/user/manipulation-llm')

import pandas as pd
import matplotlib.pyplot as plt
import pickle

run = "gemma_2b_scratchpad_prefill-12-30_16-30-00"

# Load the data directly
with open(f'notebooks/data_for_figures/{run}.pkl', 'rb') as f:
    df = pickle.load(f)

print("Columns:", df.columns.tolist())

# Get metrics available  
metrics = ['traj_rew', 'omission', 'convincing_not_to_book', 'implies_booking_succeeded', 'error_mentioning']

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Group by iteration_number and get means
agg_df = df.groupby('iteration_number')[metrics].mean().reset_index()
print("\nAggregated by iteration:")
print(agg_df.to_string())

# Rename for prettier legend
rename_map = {
    'traj_rew': 'Reward',
    'omission': 'Lying (Omission)',
    'convincing_not_to_book': 'Nudge Not Book',
    'implies_booking_succeeded': 'Implies Success',
    'error_mentioning': 'Error Mention'
}

# Plot each metric
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
for i, metric in enumerate(metrics):
    label = rename_map.get(metric, metric)
    ax.plot(agg_df['iteration_number'], agg_df[metric], marker='o', label=label, color=colors[i % len(colors)], linewidth=2)

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Booking-Assistance: Gemma 2B with Scratchpad', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xticks(agg_df['iteration_number'])

plt.tight_layout()
plt.savefig('gemma2b_booking_results.png', dpi=150)
print("\nSaved to gemma2b_booking_results.png")
