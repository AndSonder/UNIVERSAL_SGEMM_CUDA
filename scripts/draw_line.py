import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, ScalarFormatter
# Data from the table
algo = ['naive', 'global_memory_coalescing', 'shared_memory', 'blocktiling_1d', 'blocktiling_2d']
algorithm = []
for item in algo:
    algorithm += [item]*6

data = {
    'Algorithm': algorithm,
    'Matrix': ['Matrix 1', 'Matrix 2', 'Matrix 3', 'Matrix 4', 'Matrix 5', 'Matrix 6']*len(algo),
    'Time (us)': [123707, 179951, 373887, 702349, 252176, 439462,
                  8976.98, 35116.6, 65442, 79366, 28115.3, 63925.2,
                  13594.9, 42083.7, 85513.6, 114396, 40141.6, 88951.5,
                  2858.95, 4176.16, 8272.45, 10400.3, 3626.16, 8499.03,
                  2983.98, 2572.57, 5207.49, 8763.37, 3188.26, 5149.85]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Create a figure and axis
fig, ax = plt.subplots()

# Map algorithm names to unique colors
algorithm_colors = {
    'naive': 'blue',
    'global_memory_coalescing': 'green',
    'shared_memory': 'red',
    'blocktiling_1d': 'purple',
    'blocktiling_2d': 'orange'
}

# Plot each line
for algo, color in algorithm_colors.items():
    algo_data = df[df['Algorithm'] == algo]
    x = algo_data['Matrix']  # Use only 'Matrix' for x-axis labels
    y = algo_data['Time (us)']
    ax.plot(x, y, label=algo, marker='o', color=color, linestyle='-')

# Set labels and title
ax.set_xlabel('Matrix')
ax.set_ylabel('Time (us)')
ax.set_title('SGEMM Algorithms Performance')
ax.legend()

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Customize axis ticks
ax.set_xticks(df['Matrix'].unique())  # Use unique matrices for ticks
ax.set_xticklabels(sorted(df['Matrix'].unique()))  # Sort to maintain order

ax.set_yscale('log')
# Customize log-scale axis ticks
ax.yaxis.set_major_locator(LogLocator(subs=[1, 1.5, 2, 10]))

# Use ScalarFormatter for log-scale axis labels
ax.yaxis.set_major_formatter(ScalarFormatter())

# Add a legend with a shadow and a title
legend = ax.legend(title='Algorithm', loc='upper left')
legend.get_title().set_fontsize('11')

# Add a horizontal line at y=0 for better readability
ax.axhline(0, color='black', linewidth=0.5)

# Tighten layout for better appearance
plt.tight_layout()

# Save the plot to a local file (e.g., PNG format)
output_file_path = './imgs/algorithm_performance_plot.png'
plt.savefig(output_file_path, format='png', dpi=300)

# Display the saved file path
print(f"Plot saved to: {output_file_path}")

# Show the plot
plt.show()
