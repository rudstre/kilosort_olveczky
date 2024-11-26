import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from kilosort.data_tools import get_cluster_spikes, get_spike_waveforms
from datetime import datetime
from sampleToDateTime import group_spikes_by_time, sample_to_datetime
import matplotlib.dates as mdates
import logging


def process_cluster(cluster_id, cluster_info, metadata):
    # Get cluster-specific data
    ch = cluster_info.loc[cluster_info['cluster_id'] == cluster_id, 'ch'].values
    spikes = np.array(get_cluster_spikes(cluster_id, metadata.results_dir))
    spike_waveforms = np.squeeze(
        get_spike_waveforms(spikes, metadata.results_dir, chan=np.where(metadata.chanMap == ch))
    )

    # Group spikes by time intervals
    groups, unique_groups = group_spikes_by_time(spikes, metadata, metadata.th_time)

    # Filter groups based on minimum spike count
    filtered_groups = [
        group_id for group_id in unique_groups if (groups == group_id).sum() >= metadata.th_spikes
    ]
    n_filtered_groups = len(filtered_groups)

    logging.info(
        f"Cluster {cluster_id}: {n_filtered_groups} groups retained after filtering by min_spikes={metadata.th_spikes}."
    )

    if n_filtered_groups == 0:
        logging.warning(f"Cluster {cluster_id} has no groups with sufficient spikes. Skipping.")
        return

    # Compute average waveforms for each filtered group
    avg_waveforms = np.zeros((spike_waveforms.shape[0], n_filtered_groups))
    for i, group_id in enumerate(filtered_groups):
        group_indices = np.where(groups == group_id)[0]
        avg_waveforms[:, i] = spike_waveforms[:, group_indices].mean(axis=1)

    # Compute representative datetime and size for each group
    mean_datetimes = []
    sizes = []
    for group_id in filtered_groups:
        group_indices = np.where(groups == group_id)[0]
        group_spikes = spikes[group_indices]
        mean_sample = np.mean(group_spikes)
        mean_datetimes.append(sample_to_datetime(int(mean_sample), metadata))
        sizes.append(len(group_spikes))

    # Normalize sizes for plotting
    sizes_log = np.log10(np.array(sizes) + 1)
    th_spikes_log = np.log10(metadata.th_spikes + 1)
    sizes_normalized = np.interp(
        sizes_log, (th_spikes_log, sizes_log.max()), (5, 1000)
    )

    # Create plots
    palette = sns.color_palette(metadata.cmap, n_colors=len(unique_groups))
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=350, constrained_layout=True)

    # Plot 1: Averaged Waveforms
    time = np.arange(avg_waveforms.shape[0]) / 30  # Convert to ms
    for i, group_id in enumerate(filtered_groups):
        sns.lineplot(
            x=time, y=avg_waveforms[:, i], ax=axs[0], color=palette[group_id], linewidth=2.5
        )

    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"Averaged Spike Waveforms for Cluster {cluster_id}")

    # Plot 2: Spike Occurrences by Group
    for mean_datetime, size, group_id, count in zip(mean_datetimes, sizes_normalized, filtered_groups, sizes):
        axs[1].scatter(mean_datetime, 0, color=palette[group_id], s=size, label=f"Group {group_id + 1}")
        axs[1].text(
            mean_datetime,
            0.01,
            f"{count}",
            fontsize=12,
            ha="center",
            va="bottom",
            color="black",
            fontweight="bold",
        )

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1].set_xticks([datetime.combine(date, datetime.min.time()) for date in metadata.unique_dates])
    plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_xlabel("Date")
    axs[1].set_yticks([])
    axs[1].set_ylabel("Spike")
    axs[1].set_title("Spike Occurrences Over Time by Group")

    # Add colorbar for group progression
    sm = plt.cm.ScalarMappable(cmap=metadata.cmap, norm=plt.Normalize(vmin=0, vmax=len(unique_groups) - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axs, label="Group Progression")

    plt.show()
