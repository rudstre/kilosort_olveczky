import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from kilosort.data_tools import get_cluster_spikes
from datetime import datetime
from sampleToDateTime import sample_to_datetime
from pathlib import Path
import click
import shlex
import warnings
from typing import Dict, Optional, Any
from collections import Counter

# Suppress matplotlib and seaborn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def dewhiten_data(data, Winv, is_template=True, peak_channel=None, convert_to_uV=True):
    """
    Dewhiten data using the inverse whitening matrix.

    Args:
        data: Data to dewhiten (2D template or 1D amplitudes)
        Winv: Inverse whitening matrix
        is_template: True if template data, False for amplitudes
        peak_channel: Required if is_template is False
        convert_to_uV: Whether to scale result to microvolts

    Returns:
        Dewhitened data array
    """
    uV_factor = 0.195 if convert_to_uV else 1.0
    if is_template:
        return (Winv @ data.T).T * uV_factor
    else:
        if peak_channel is None:
            raise ValueError("Peak channel must be provided for amplitude dewhitening")
        weights = Winv[peak_channel, :]
        factor = weights[peak_channel]
        return data * factor * uV_factor


def prepare_cluster_data(cluster_id: int, metadata, min_amplitude: float = 0) -> Optional[Dict[str, Any]]:
    """
    Prepare all data needed for plotting a cluster's waveforms and metrics.
    
    Args:
        cluster_id: The ID of the cluster to analyze
        metadata: Metadata object with recording information
        min_amplitude: Minimum amplitude threshold in μV
        
    Returns:
        Dictionary containing all data needed for plotting, or None if cluster should be filtered out
    """
    # Extract spikes
    spikes = np.array(get_cluster_spikes(cluster_id, metadata.results_dir))
    if spikes.ndim > 1 and spikes.shape[0] > 0:
        spikes = spikes[0]
    if len(spikes) == 0:
        return None

    # Load template and whitening data
    base = metadata.results_dir
    tpl_file = base / 'templates.npy'
    shank_file = base / 'channel_shanks.npy'
    inv_file = base / 'whitening_mat_inv.npy'
    if not (tpl_file.exists() and shank_file.exists() and inv_file.exists()):
        return None

    templates = np.load(tpl_file)
    shanks = np.load(shank_file)
    Winv = np.load(inv_file)
    template_white = templates[cluster_id]
    template = dewhiten_data(template_white, Winv, is_template=True)

    # Determine best channel and tetrode channels
    chan_amps = np.max(np.abs(template), axis=0)
    best_idx = np.argmax(chan_amps)
    wave = template[:, best_idx]
    template_p2p = wave.max() - wave.min()
    template_abs_max = np.max(np.abs(wave))
    tetrode = shanks[best_idx]
    chans = np.where(shanks == tetrode)[0]
    if len(chans) > 4:
        dists = np.abs(chans - best_idx)
        chans = chans[np.argsort(dists)[:4]]
    chans = sorted(chans[:4])

    # Determine the sign of the template
    template_peak_sign = np.sign(wave.min() if abs(wave.min()) > abs(wave.max()) else wave.max())

    # Load spike amplitudes if available
    amp_file = base / 'amplitudes.npy'
    clust_file = base / 'spike_clusters.npy'
    if amp_file.exists() and clust_file.exists():
        all_amps = np.load(amp_file)
        all_clust = np.load(clust_file)
        mask = (all_clust == cluster_id)
        amps_raw = dewhiten_data(all_amps[mask], Winv, is_template=False, peak_channel=best_idx)
        
        # Apply sign to ensure amplitudes match the template polarity
        amps_raw = amps_raw * template_peak_sign
        
        # Match amplitude scale to template
        amp_med = np.median(abs(amps_raw))
        template_peak = template_abs_max
        scaling_methods = [
            ("Current method", amps_raw),
            ("Scale to template norm", amps_raw * (template_peak / amp_med)),
            ("Peak-to-peak match", amps_raw * (template_p2p / (2 * amp_med)))
        ]
        for name, scaled in scaling_methods:
            ratio = np.median(abs(scaled)) / template_peak
            if 0.9 <= ratio <= 1.1:
                amps = scaled if name != "Current method" else amps_raw
                break
        else:
            amps = amps_raw
        
        # Check if median amplitude meets minimum threshold
        if np.median(abs(amps)) < min_amplitude:
            return None
            
        n = min(len(amps), len(spikes))
        amps = amps[:n]
        spikes = spikes[:n]
    else:
        # When amplitudes aren't available, use template amplitude with sign
        amps = np.full(len(spikes), template_peak_sign * template_abs_max)
        
        # Check if amplitude meets minimum threshold
        if template_abs_max < min_amplitude:
            return None

    # Convert spike times to datetime
    vec = np.vectorize(lambda s: sample_to_datetime(s, metadata))
    try:
        times = vec(spikes)
    except:
        times = np.array([datetime.now()] * len(spikes))

    # Compute ISI if possible
    isi_ms = None
    if len(spikes) > 1:
        sorted_sp = np.sort(spikes)
        isi = np.diff(sorted_sp)
        isi_ms = isi / (metadata.samples_per_second / 1000.0)
        log_isi = np.log10(isi_ms) if len(isi_ms) > 0 else None
        isi_violations = np.sum(isi_ms < 1.0) if isi_ms is not None else 0
        isi_violation_pct = (isi_violations / len(isi_ms) * 100) if len(isi_ms) > 0 else 0
    else:
        log_isi = None
        isi_violations = 0
        isi_violation_pct = 0

    # Time vector for waveform
    samples_per_ms = metadata.samples_per_second / 1000
    t = np.arange(template.shape[0]) / samples_per_ms

    # Prepare amplitude plot data
    if len(np.unique([dt.date() for dt in times])) <= 1:
        # Single day recording - use seconds since start of day
        secs = np.array([(dt - datetime.combine(dt.date(), datetime.min.time())).total_seconds() for dt in times])
        x_time = secs
        time_format = 'seconds'
    else:
        # Multi-day recording - use dates
        x_time = times
        time_format = 'dates'

    # Return all calculated data
    return {
        'cluster_id': cluster_id,
        'tetrode': int(tetrode),
        'channels': chans,
        'template': template,
        'template_abs_max': template_abs_max,
        'template_p2p': template_p2p,
        'template_peak_sign': template_peak_sign,
        't': t,
        'spikes': spikes,
        'times': times,
        'x_time': x_time,
        'time_format': time_format,
        'amps': amps,
        'isi_ms': isi_ms,
        'log_isi': log_isi,
        'isi_violations': isi_violations,
        'isi_violation_pct': isi_violation_pct,
        'n_spikes': len(spikes)
    }


def plot_single_waveform_with_amplitude(cluster_id, cluster_info, metadata, min_amplitude=0):
    """
    Plot template waveforms, spike amplitude over time, and ISI distribution using seaborn
    
    Args:
        cluster_id: The ID of the cluster to plot
        cluster_info: DataFrame containing cluster information
        metadata: Metadata object with recording information
        min_amplitude: Minimum amplitude threshold in μV (clusters with median amplitude below this will return None)
        
    Returns:
        A matplotlib.figure.Figure or None
    """
    # Prepare data for plotting
    data = prepare_cluster_data(cluster_id, metadata, min_amplitude)
    if data is None:
        return None
    
    # Create figure and GridSpec
    return create_cluster_figure(data)


def create_cluster_figure(data: Dict[str, Any]) -> plt.Figure:
    """
    Create a figure with waveforms, amplitude, and ISI plots for a cluster.
    
    Args:
        data: Dictionary of data from prepare_cluster_data
        
    Returns:
        matplotlib.figure.Figure
    """
    # Setup figure with constrained_layout enabled
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    
    # Create a figure-level GridSpec with 3 rows
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 1])
    
    # Top row will contain channel plots
    chans = data['channels']
    channel_grid = gs[0].subgridspec(nrows=1, ncols=len(chans), wspace=0.15)
    
    # Plot waveforms with proper subplot positioning
    template = data['template']
    y_min = template.min() * 1.1
    y_max = template.max() * 1.1
    t = data['t']
    
    for i, ch_idx in enumerate(chans):
        ax = fig.add_subplot(channel_grid[i])
        
        # Plot the waveform
        sns.lineplot(x=t, y=template[:, ch_idx], ax=ax, linewidth=2.5)
        ax.axhline(0, linestyle='-', alpha=0.5)
        for tm in range(0, int(t.max())+1, 10):
            ax.axvline(tm, linestyle='--', alpha=0.3)
        ax.set_title(f'Ch {ch_idx}', fontsize=10)
        
        # Only show y-axis on first plot
        if i == 0:
            ax.set_ylabel('Amplitude (μV)')
        else:
            ax.set_yticks([])
            
        # Keep x-axis labels and ticks consistent
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(0, t.max())
        ax.set_ylim(y_min, y_max)
        sns.despine(ax=ax)

    # Plot spike amplitudes
    amp_ax = fig.add_subplot(gs[1])
    
    if data['time_format'] == 'seconds':
        # Single day recording
        sns.scatterplot(x=data['x_time'], y=data['amps'], ax=amp_ax, s=10, alpha=0.6)
        amp_ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"))
        amp_ax.set_xlabel('Time of Day (HH:MM:SS)')
    else:
        # Multi-day recording
        sns.scatterplot(x=data['times'], y=data['amps'], ax=amp_ax, s=10, alpha=0.6)
        amp_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        amp_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.setp(amp_ax.get_xticklabels(), rotation=45, ha='right')
        amp_ax.set_xlabel('Date')
        
    # Restrict y-axis to bulk of data while preserving sign
    amps = data['amps']
    template_abs_max = data['template_abs_max']
    med_amp = np.median(amps)
    p5, p95 = np.percentile(amps, [5, 95])
    amp_range = max(p95 - p5, 1)
    
    # Make sure we include zero in the plot
    if med_amp > 0:
        lower = min(-template_abs_max * 0.1, med_amp - 2 * amp_range)
        upper = med_amp + 2 * amp_range
    else:
        lower = med_amp - 2 * amp_range
        upper = max(template_abs_max * 0.1, med_amp + 2 * amp_range)
    amp_ax.set_ylim(lower, upper)
    amp_ax.axhline(0, linestyle='-', alpha=0.3, color='gray')  # Add zero line
    amp_ax.set_ylabel('Amplitude (μV)')
    amp_ax.set_title(f'Spike Amplitude Over Time (n={data["n_spikes"]})', pad=12)

    # Plot ISI distribution
    isi_ax = fig.add_subplot(gs[2])
    
    if data['log_isi'] is not None and len(data['log_isi']) > 0:
        try:
            sns.kdeplot(x=data['log_isi'], fill=True, ax=isi_ax)
        except:
            sns.histplot(data['log_isi'], bins=50, stat='density', ax=isi_ax)
        for v in [0, 1, 2, 3]:
            isi_ax.axvline(v, linestyle='--', alpha=0.5)
        isi_ax.set_xlabel('ISI (log10(ms))')
        isi_ax.set_ylabel('Density')
        
        if data['isi_violations'] > 0:
            isi_ax.text(0.05, 0.95, 
                      f'Refractory violations: {data["isi_violations"]} ({data["isi_violation_pct"]:.2f}%)',
                      transform=isi_ax.transAxes, va='top', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        isi_ax.text(0.5, 0.5, 'Not enough spikes to calculate ISI',
                    transform=isi_ax.transAxes, ha='center', va='center')
    isi_ax.set_title('Inter-Spike Interval Distribution', pad=12)

    # Add title
    fig.suptitle(f'Tetrode {data["tetrode"]} Waveforms for Cluster {data["cluster_id"]}', fontsize=16)
    
    return fig


def create_summary_page(cluster_data_list, min_amplitude):
    """
    Create a summary page showing clusters per tetrode
    
    Args:
        cluster_data_list: List of dictionaries with cluster data from prepare_cluster_data
        min_amplitude: Minimum amplitude threshold that was applied
        
    Returns:
        matplotlib.figure.Figure
    """
    # Use the same figure size as cluster figures for consistency
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    
    # Count clusters per tetrode
    tetrode_counts = Counter([data['tetrode'] for data in cluster_data_list])
    
    # Find all possible tetrode numbers from the data
    all_tetrodes = set([data['tetrode'] for data in cluster_data_list])
    
    # Calculate recording time range and duration
    if cluster_data_list:
        # Collect all spike times from all clusters
        all_times = []
        for data in cluster_data_list:
            all_times.extend(data['times'])
        
        # Get min and max dates/times
        if all_times:
            start_time = min(all_times)
            end_time = max(all_times)
            duration_sec = (end_time - start_time).total_seconds()
            
            # Format duration as days, hours, minutes, seconds
            days = int(duration_sec // (24 * 3600))
            remaining = duration_sec % (24 * 3600)
            hours = int(remaining // 3600)
            remaining %= 3600
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            
            if days > 0:
                duration_str = f"{days}d {hours}h {minutes}m {seconds}s"
            elif hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            else:
                duration_str = f"{minutes}m {seconds}s"
            
            time_range_str = f"Recording period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            duration_info = f"Recording duration: {duration_str}"
        else:
            time_range_str = "Recording period: Unknown"
            duration_info = "Recording duration: Unknown"
    else:
        time_range_str = "Recording period: No data available"
        duration_info = "Recording duration: Unknown"
    
    # Create a range from min to max tetrode numbers to ensure all are shown
    if all_tetrodes:
        min_tetrode = min(all_tetrodes)
        max_tetrode = max(all_tetrodes)
        # Show at least tetrodes 1-16 or extend to max if higher
        max_tetrode = max(16, max_tetrode)
        tetrodes = list(range(min_tetrode, max_tetrode + 1))
    else:
        # Default range if no tetrodes found
        tetrodes = list(range(1, 17))
    
    # Get counts for all tetrodes (0 for those without clusters)
    counts = [tetrode_counts.get(t, 0) for t in tetrodes]
    
    # Create the bar chart
    ax = fig.add_subplot(111)
    bars = ax.bar(tetrodes, counts, color='steelblue', alpha=0.7)
    
    # Add count labels on top of bars for non-zero counts
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Tetrode Number', fontsize=14)
    ax.set_ylabel('Number of Clusters', fontsize=14)
    ax.set_xticks(tetrodes)
    
    # Improved summary title
    title = "Kilosort Cluster Summary"
    ax.set_title(title, fontsize=18, pad=20)
    
    # Ensure y-axis has enough room for labels
    if max(counts) > 0:
        ax.set_ylim(0, max(counts) * 1.15)  # Add 15% padding on top
    
    # Add a text box with summary info
    # Format with clear section headers and more readable design
    textstr = (
        "RECORDING SUMMARY\n"
        "---------------------------------------------\n"
        f"• Total Clusters: {len(cluster_data_list)}\n"
        f"• Total Tetrodes with Units: {len(tetrode_counts)}\n"
        f"• Min Amplitude Threshold: {min_amplitude} µV\n\n"
        "TIME INFORMATION\n"
        "---------------------------------------------\n"
        f"• {time_range_str.replace('Recording period: ', '')}\n"
        f"• {duration_info.replace('Recording duration: ', '')}\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    
    # Create a more readable box with less transparency and better contrast
    props = dict(
        boxstyle='round,pad=1',
        facecolor='lightgoldenrodyellow',
        alpha=0.9,
        edgecolor='darkgray',
        linewidth=2
    )
    
    # Position in upper right with larger font
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, 
            fontsize=14, fontweight='normal', linespacing=1.3,
            verticalalignment='top', horizontalalignment='right', 
            bbox=props)
    
    # Some styling for the figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig


if __name__ == "__main__":
    import logging
    import seaborn as sns
    from pathlib import Path
    import pandas as pd
    from tqdm import tqdm
    from kilosort.io import load_ops
    from sampleToDateTime import load_metadata
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Use non-interactive backend to avoid warnings
    plt.switch_backend('Agg')
    
    # apply seaborn style globally
    sns.set_style("whitegrid")

    # Configure logging to be less verbose
    logging.basicConfig(
        level=logging.ERROR,  # Changed from WARNING to ERROR
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    @click.command()
    @click.option('--results-dir', '-d', type=click.Path(file_okay=False, resolve_path=True),
                  help='Path to the Kilosort results directory')
    @click.option('--min-amplitude', '-a', type=float, default=None,
                  help='Minimum amplitude threshold in μV (default: 0, no filtering)')
    @click.option('--quiet', '-q', is_flag=True, help='Reduce terminal output')
    def main(results_dir, min_amplitude, quiet):
        """Generate waveform plots with amplitude filtering"""
        
        # Interactive prompt if results directory not provided
        if results_dir is None:
            while True:
                path_input = click.prompt(
                    "Enter path to Kilosort results directory",
                    type=str
                )
                
                # Use shlex to properly handle quoted strings
                try:
                    # This correctly handles quotes, escapes, etc.
                    unquoted_path = shlex.split(path_input)[0]
                    path = Path(unquoted_path).expanduser()
                    if path.exists():
                        results_dir = path.resolve()
                        break
                    else:
                        click.echo(f"Error: Directory {path} does not exist. Please enter a valid path.")
                except (IndexError, ValueError) as e:
                    click.echo(f"Error parsing path: {e}. Please enter a valid path.")
        else:
            # Convert to Path object
            results_dir = Path(results_dir)
        
        # Prompt for minimum amplitude if not provided via command line
        if min_amplitude is None:
            min_amplitude = click.prompt(
                "Enter minimum amplitude threshold in μV",
                type=float, default=0.0
            )
        
        click.echo(f"Loading cluster information and metadata from {results_dir}...")

        try:
            ops = load_ops(results_dir / "ops.npy")
            metadata = load_metadata(results_dir, ops)

            cluster_info = pd.read_csv(
                results_dir / "cluster_group.tsv",
                sep="\t"
            )
        except FileNotFoundError as e:
            click.echo(f"Error: Required files not found in {results_dir}")
            click.echo(f"Detailed error: {e}")
            sys.exit(1)

        cluster_ids = cluster_info.loc[
            cluster_info['KSLabel'] != 'noise',
            'cluster_id'
        ].tolist()

        suffix = f"_min{min_amplitude}uV" if min_amplitude > 0 else ""
        pdf_path = results_dir / f"cluster_waveforms{suffix}.pdf"
        click.echo(f"Generating PDF at {pdf_path}")
        click.echo(f"Minimum amplitude threshold: {min_amplitude} μV")
        click.echo(f"Processing {len(cluster_ids)} clusters...")

        # Process all clusters first to collect data for summary page
        all_cluster_data = []
        # Use disable=quiet to optionally hide the progress bar
        for i, clust_id in enumerate(tqdm(cluster_ids, desc="Processing clusters", disable=quiet)):
            try:
                data = prepare_cluster_data(
                    clust_id, metadata, min_amplitude=min_amplitude
                )
                if data is not None:
                    all_cluster_data.append(data)
                
                # Print a simple progress indicator when quiet mode is enabled
                if quiet and (i+1) % 10 == 0:
                    click.echo(f"Processed {i+1}/{len(cluster_ids)} clusters")
                    
            except Exception as e:
                if not quiet:
                    logging.error(f"Error on cluster {clust_id}: {e}")

        click.echo(f"Creating PDF with {len(all_cluster_data)} clusters that met amplitude criteria...")

        with PdfPages(pdf_path) as pdf:
            # Add summary page as the first page
            summary_fig = create_summary_page(all_cluster_data, min_amplitude)
            pdf.savefig(summary_fig)
            plt.close(summary_fig)
            
            # Add individual cluster pages
            for i, data in enumerate(tqdm(all_cluster_data, desc="Generating figures", disable=quiet), 1):
                try:
                    fig = create_cluster_figure(data)
                    pdf.savefig(fig)
                    plt.close(fig)
                    
                    # Print progress periodically in quiet mode
                    if quiet and i % 10 == 0:
                        click.echo(f"Created {i}/{len(all_cluster_data)} figures")
                except Exception as e:
                    if not quiet:
                        logging.error(f"Error creating figure for cluster {data['cluster_id']}: {e}")

        click.echo(f"PDF generation complete – saved to {pdf_path}")
        click.echo(f"Included {len(all_cluster_data)}/{len(cluster_ids)} clusters that met amplitude criteria")

    main()