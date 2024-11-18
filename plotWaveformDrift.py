import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from kilosort.io import load_ops
from kilosort.data_tools import (
    mean_waveform, cluster_templates, get_good_cluster, get_labels, get_cluster_spikes,
    get_spike_waveforms, get_best_channel
)

# Set Seaborn theme and color palette
sns.set_theme(style="whitegrid")

# File paths
results_dir = Path("C:/Users/Rudy/Documents/recordings/kilosort_out")
cluster_info_path = results_dir / "cluster_info.tsv"
new_cluster_id = 23  # The new Phy-assigned `cluster_id` you want to plot

# Load the cluster mapping with pandas, which handles headers and data types
cluster_info = pd.read_csv(cluster_info_path, sep='\t')

# Assume that the first column is the original Kilosort ID and the second is the new Phy ID
# Adjust this if the columns are reversed or named differently
cluster_map = dict(zip(cluster_info.cluster_id.iloc[:, 1], cluster_info.cluster_id.iloc[:, 0]))

# Get the original Kilosort ID for the specified new Phy cluster ID
original_cluster_id = cluster_map.get(new_cluster_id)

# Load spike clusters and select spikes for the specified (original) cluster ID
spike_clusters = np.load(results_dir / "spike_clusters.npy")
spikes = np.array([s for s, c in zip(get_cluster_spikes(original_cluster_id, results_dir), spike_clusters) if c == original_cluster_id])
waves = get_spike_waveforms(spikes, results_dir, chan=get_best_channel(original_cluster_id, results_dir))

# Parameters
every = 500
num_spikes = waves.shape[1]
num_samples = waves.shape[0]
n_groups = num_spikes // every  # Calculate the number of groups based on total spikes

# Average spike waveforms
averaged_spikes = waves[:, :n_groups * every].reshape(num_samples, n_groups, every).mean(axis=2)
n_averaged_spikes = n_groups  # Set this based on the actual number of groups

# Set up the color palette
palette = sns.color_palette("viridis", n_colors=n_averaged_spikes)

# Load total samples from spike_times.npy for recording duration
spike_times = np.load(results_dir / 'spike_times.npy')  # Load spike times in samples
total_samples = spike_times.max()  # Max sample as recording end
total_time_minutes = total_samples / 1.8e6  # Convert to minutes (assuming 30kHz)

# Create a two-row plot layout
fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=350, gridspec_kw={'height_ratios': [1, 0.5]})

# Plot 1: Averaged Spike Waveform Drift Over Time
time = np.arange(averaged_spikes.shape[0])
for i in range(n_averaged_spikes):
    sns.lineplot(x=time, y=averaged_spikes[:, i], ax=axs[0], color=palette[i], linewidth=2.5)

# Add color bar to the averaged waveform plot
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_averaged_spikes))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs[0], ticks=[0, n_averaged_spikes])
cbar.ax.set_yticklabels(['Early', 'Late'])
cbar.set_label('Spike Order')

# Add labels and title for the averaged waveform plot
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Averaged Spike Waveform Drift Over Time')

# Plot 2: Raster Plot of Spike Occurrences
# Convert spike times to minutes based on 30kHz sampling rate
spike_times_min = spikes / 1.8e6  # Convert samples to minutes
spike_groups = np.arange(len(spikes)) // every  # Assign each spike to a group based on index

# Plot each spike at its actual timestamp with the color of its group
for group_idx in range(n_averaged_spikes):
    # Select spikes that belong to the current group
    group_spikes = spike_times_min[spike_groups == group_idx]
    axs[1].scatter(group_spikes, np.zeros_like(group_spikes), color=palette[group_idx], s=10, label=f'Group {group_idx+1}')

# Formatting the raster plot
axs[1].set_ylim(-0.1, 0.1)
axs[1].set_xlim(0, total_time_minutes)  # Set x-axis to cover the full recording duration
axs[1].set_xlabel('Time (minutes)')
axs[1].set_yticks([])
axs[1].set_ylabel('Spike')
axs[1].set_title('Spike Occurrences Over Time by Group')

plt.tight_layout()
plt.show()
