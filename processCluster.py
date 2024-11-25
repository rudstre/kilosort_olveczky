import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kilosort.data_tools import get_cluster_spikes, get_spike_waveforms
from sampleToDateTime import samples_to_relative_seconds
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import logging

def process_cluster(cluster_id, cluster_info, metadata):
    # Get cluster-specific data
    ch = cluster_info.loc[cluster_info['cluster_id'] == cluster_id, 'ch'].values
    spikes = np.array(get_cluster_spikes(cluster_id, metadata.results_dir))
    spike_waveforms = np.squeeze(
        get_spike_waveforms(spikes, metadata.results_dir, chan=np.where(metadata.chanMap == ch)))
    spike_times_sec = samples_to_relative_seconds(spikes, metadata.data)

    # Group spikes by time intervals
    groups = [0]
    for i in range(1, len(spike_times_sec)):
        if spike_times_sec[i] - spike_times_sec[i - 1] > metadata.th_time:
            groups.append(groups[-1] + 1)
        else:
            groups.append(groups[-1])
    groups = np.array(groups)
    unique_groups = np.unique(groups)

    # Filter out groups with fewer spikes than the threshold
    group_spike_counts = {group_id: (groups == group_id).sum() for group_id in unique_groups}
    filtered_groups = [group_id for group_id, count in group_spike_counts.items() if count >= metadata.th_spikes]
    n_groups = len(filtered_groups)
    logging.info(f"Cluster {cluster_id}: {n_groups} groups retained after filtering by min_spikes={metadata.th_spikes}.")

    if n_groups == 0:
        logging.warning(f"Cluster {cluster_id} has no groups with sufficient spikes. Skipping.")
        return

    # Compute average waveforms for each filtered group
    waveform_length = spike_waveforms.shape[0]
    avg_waveforms = np.zeros((waveform_length, n_groups))
    for i, group_id in enumerate(filtered_groups):
        group_indices = np.where(groups == group_id)[0]
        avg_waveforms[:, i] = spike_waveforms[:, group_indices].mean(axis=1)

    # Create plots
    palette = sns.color_palette(metadata.cmap, n_colors=n_groups)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=350, gridspec_kw={'height_ratios': [1, 0.5]},
                            constrained_layout=True)

    # Plot 1: Averaged Waveform
    time = np.arange(avg_waveforms.shape[0]) / 30  # Convert to ms
    for i in range(n_groups):
        sns.lineplot(x=time, y=avg_waveforms[:, i], ax=axs[0], color=palette[i], linewidth=2.5)

    # Plot 2: Representative Dots
    mean_datetimes = []
    sizes = []
    for group_id in filtered_groups:
        group_spikes = spike_times_sec[groups == group_id]
        group_datetimes = [metadata.tstart + timedelta(seconds=spike) for spike in group_spikes]
        timestamps = np.array([dt.timestamp() for dt in group_datetimes])
        mean_datetime = datetime.fromtimestamp(np.mean(timestamps))
        mean_datetimes.append(mean_datetime)
        sizes.append(len(group_spikes))

    sizes_log = np.log10(np.array(sizes) + 1)
    th_spikes_log = np.log10(metadata.th_spikes + 1)
    min_size, max_size = 5, 1000

    sizes_normalized = min_size + (sizes_log - th_spikes_log) / (sizes_log.max() - th_spikes_log) * (
                max_size - min_size)

    for mean_datetime, size, group_id, count in zip(mean_datetimes, sizes_normalized, filtered_groups, sizes):
        axs[1].scatter(mean_datetime, 0, color=palette[group_id], s=size, label=f'Group {group_id + 1}')
        radius = np.sqrt(size) / 2
        axs[1].text(mean_datetime, 0 + radius * 0.01, f'{count}', fontsize=12, ha='center', va='bottom', color='black',
                    fontweight='bold')

    mean_datetimes = np.array(mean_datetimes)
    sm = plt.cm.ScalarMappable(cmap=metadata.cmap, norm=plt.Normalize(vmin=0, vmax=n_groups - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, ticks=[0, n_groups - 1])
    cbar.ax.set_yticklabels(['Early', 'Late'])
    cbar.set_label('Group (by Time)')

    axs[0].set_xlabel('Time (samples)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f'Averaged Spike Waveforms for Cluster {cluster_id}')

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
