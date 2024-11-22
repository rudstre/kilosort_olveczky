import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from kilosort.io import load_ops
from kilosort.data_tools import (
    get_best_channel, get_cluster_spikes, get_spike_waveforms, cluster_templates
)
from sampleToDateTime import samples_to_relative_seconds, load_metadata
from datetime import timedelta, datetime
from dateutil import parser
import matplotlib.dates as mdates

cluster_id = 83  # Replace with desired cluster ID

# Set Seaborn theme and color palette
sns.set_theme(style="whitegrid")

# File paths
results_dir = Path("C:/Users/Rudy/Documents/kilosort_out/kilosort4")
cluster_info_path = results_dir / "cluster_info.tsv"
cluster_info = pd.read_csv(cluster_info_path, sep='\t')

metadata_path = results_dir / "../recording_metadata.json"
metadata = load_metadata(metadata_path)
recording_start = parser.parse(metadata[0]["datetime"])
recording_end = parser.parse(metadata[-1]["datetime"])

ops = load_ops(results_dir / 'ops.npy')
chanMap = ops['chanMap']

# Load spike clusters and waveforms
ch = cluster_info.loc[cluster_info['cluster_id'] == cluster_id, 'ch'].values

spikes = np.array(get_cluster_spikes(cluster_id, results_dir))
spike_waveforms = np.squeeze(get_spike_waveforms(spikes, results_dir, chan=np.where(chanMap == ch)))

# Convert spike samples to absolute timestamps
spike_times_sec = samples_to_relative_seconds(spikes, metadata)

# **Step 1: Group spikes by 1-hour intervals**
# Define the 1-hour threshold in seconds
threshold_seconds = 3600  # 1 hour
groups = [0]  # Start with the first spike in group 0
for i in range(1, len(spike_times_sec)):
    if spike_times_sec[i] - spike_times_sec[i - 1] > threshold_seconds:
        groups.append(groups[-1] + 1)  # Start a new group
    else:
        groups.append(groups[-1])  # Same group as the previous spike

groups = np.array(groups)  # Convert to NumPy array
unique_groups = np.unique(groups)  # Unique group IDs
n_groups = len(unique_groups)

# **Step 2: Compute average waveforms for each group**
waveform_length = spike_waveforms.shape[0]  # Number of timepoints per waveform
avg_waveforms = np.zeros((waveform_length, n_groups))  # Initialize a 2D NumPy array
for i, group_id in enumerate(unique_groups):
    group_indices = np.where(groups == group_id)[0]
    avg_waveforms[:, i] = spike_waveforms[:, group_indices].mean(axis=1)

# Set up the color palette
palette = sns.color_palette("viridis", n_colors=n_groups)

fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=350, gridspec_kw={'height_ratios': [1, 0.5]},constrained_layout=True)

# Plot 1: Averaged Spike Waveform Drift Over Time
time = np.arange(avg_waveforms.shape[0]) / 30 # convert to ms
for i in range(n_groups):
    sns.lineplot(x=time, y=avg_waveforms[:, i], ax=axs[0], color=palette[i], linewidth=2.5)

# Plot 2: Representative Dots for Each Group
mean_datetimes = []
sizes = []  # Sizes of the dots (proportional to spike counts)

for group_id in unique_groups:
    group_spikes = spike_times_sec[groups == group_id]
    group_datetimes = [recording_start + timedelta(seconds=spike) for spike in group_spikes]

    # Compute mean datetime for the group
    timestamps = np.array([dt.timestamp() for dt in group_datetimes])
    mean_datetime = datetime.fromtimestamp(np.mean(timestamps))
    mean_datetimes.append(mean_datetime)

    # Compute size proportional to spike count
    sizes.append(len(group_spikes))  # Number of spikes in the group

sizes_log = np.log10(np.array(sizes) + 1)  # Add 1 to handle zeros safely

# Normalize log sizes to a range (e.g., between 50 and 500)
min_size, max_size = 5, 1000
sizes_normalized = min_size + (sizes_log - sizes_log.min()) / (sizes_log.max() - sizes_log.min()) * (max_size - min_size)

# Plot representative dots
for mean_datetime, size, group_id, count in zip(mean_datetimes, sizes_normalized, unique_groups, sizes):
    axs[1].scatter(mean_datetime, 0, color=palette[group_id], s=size, label=f'Group {group_id + 1}')
    # Add text inside or next to the dot
    radius = np.sqrt(size) / 2
    axs[1].text(
        mean_datetime,  # X-coordinate (same as the dot's x position)
        0 + radius * 0.01,  # Y-coordinate (same as the dot's y position)
        f'{count}',  # Text to display (number of spikes)
        fontsize=12,  # Font size
        ha='center',
        va='bottom',
        color='black',  # Text color
        fontweight='bold'
    )
mean_datetimes = np.array(mean_datetimes)

# Add color bar to the averaged waveform plot
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_groups - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, ticks=[0, n_groups - 1])
cbar.ax.set_yticklabels(['Early', 'Late'])
cbar.set_label('Group (by Time)')

# Formatting the raster plot
# Add labels and title for the averaged waveform plot
axs[0].set_xlabel('Time (samples)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Averaged Spike Waveforms for Each Session')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
axs[1].set_ylim(-0.5, 0.5)
axs[1].set_xlabel('Time')
axs[1].set_xticks(mean_datetimes)
axs[1].set_yticks([])
axs[1].set_ylabel('Spike')
axs[1].set_title('Spike Occurrences Over Time by Group')

plt.show()
