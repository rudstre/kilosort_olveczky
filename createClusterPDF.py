import warnings
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from kilosort.data_tools import get_cluster_spikes
from sampleToDateTime import sample_to_datetime

# Suppress matplotlib warnings from external libraries
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Scaling factor: convert kilosort units to microvolts
UV_FACTOR = 0.195


def dewhiten_data(
    data: np.ndarray,
    Winv: np.ndarray,
    is_template: bool = True,
    peak_channel: Optional[int] = None,
    convert_to_uV: bool = True
) -> np.ndarray:
    """
    Dewhiten data using the inverse whitening matrix.

    Args:
        data: Template (2D) or amplitudes (1D) array
        Winv: Inverse whitening matrix
        is_template: True for template data, False for amplitudes
        peak_channel: Required channel index when dewhitening amplitudes
        convert_to_uV: Scale result to microvolts if True

    Returns:
        Dewhitened data array
    """
    scale = UV_FACTOR if convert_to_uV else 1.0

    if is_template:
        # Apply full inverse whitening to the waveform template
        return (Winv @ data.T).T * scale

    # Amplitude dewhitening: use self-weight factor
    if peak_channel is None:
        raise ValueError("Peak channel must be provided for amplitude dewhitening")

    weights = Winv[peak_channel, :]
    factor = weights[peak_channel]
    return data * factor * scale


def prepare_cluster_data(
    cluster_id: int,
    metadata,
    min_amplitude: float = 0
) -> Optional[Dict[str, Any]]:
    """
    Load and compute all necessary metrics for a single cluster.

    Filters out clusters with no spikes or below amplitude threshold.
    """
    # --- Load spike times ---
    spikes = np.array(get_cluster_spikes(cluster_id, metadata.results_dir))
    if spikes.ndim > 1 and spikes.shape[0] > 0:
        spikes = spikes[0]
    if spikes.size == 0:
        return None

    # --- Load Kilosort outputs ---
    base = metadata.results_dir
    tpl_path = base / 'templates.npy'
    shank_path = base / 'channel_shanks.npy'
    inv_path = base / 'whitening_mat_inv.npy'
    if not (tpl_path.exists() and shank_path.exists() and inv_path.exists()):
        return None

    templates = np.load(tpl_path)
    shanks = np.load(shank_path)
    Winv = np.load(inv_path)
    raw_template = templates[cluster_id]

    # --- Dewhiten to get waveform in microvolts ---
    template = dewhiten_data(raw_template, Winv, is_template=True)

    # --- Identify best channel and waveform metrics ---
    chan_amplitudes = np.max(np.abs(template), axis=0)
    best_idx = int(np.argmax(chan_amplitudes))
    wave = template[:, best_idx]
    peak_sign = np.sign(wave.min() if abs(wave.min()) > abs(wave.max()) else wave.max())
    p2p = wave.max() - wave.min()
    abs_max = np.max(np.abs(wave))

    # --- Filter by template amplitude threshold ---
    if min_amplitude > 0 and abs_max < min_amplitude:
        print(f"Cluster {cluster_id} excluded: peak {abs_max:.2f}µV < threshold {min_amplitude:.2f}µV")
        return None

    # --- Determine tetrode and channel group ---
    tetrode = int(shanks[best_idx])
    all_chans = np.where(shanks == tetrode)[0]
    # Select four closest channels to peak
    if all_chans.size > 4:
        distances = np.abs(all_chans - best_idx)
        all_chans = all_chans[np.argsort(distances)[:4]]
    channels = sorted(all_chans.tolist())

    # --- Load and dewhiten spike amplitudes ---
    try:
        amps_raw = np.load(base / 'amplitudes.npy')
        clusters = np.load(base / 'spike_clusters.npy')
        mask = (clusters == cluster_id)
        amps = dewhiten_data(amps_raw[mask], Winv, is_template=False, peak_channel=best_idx)
        amps *= peak_sign  # Match template polarity
        # Align arrays
        n = min(amps.size, spikes.size)
        amps = amps[:n]
        spikes = spikes[:n]
    except Exception as e:
        print(f"Warning: using template amplitude for cluster {cluster_id} ({e})")
        amps = np.full(spikes.size, peak_sign * abs_max)

    # --- Convert spike times to datetime objects ---
    to_dt = np.vectorize(lambda s: sample_to_datetime(s, metadata))
    try:
        times = to_dt(spikes)
    except:
        times = np.array([datetime.now()] * spikes.size)

    # --- Compute inter-spike intervals (ISI) in ms ---
    isi_ms, log_isi, violations, vio_pct = None, None, 0, 0
    if spikes.size > 1:
        sorted_sp = np.sort(spikes)
        diffs = np.diff(sorted_sp)
        isi_ms = diffs / (metadata.samples_per_second / 1000)
        if isi_ms.size:
            log_isi = np.log10(isi_ms)
            violations = int(np.sum(isi_ms < 1.0))
            vio_pct = (violations / isi_ms.size) * 100

    # --- Prepare time axis for amplitude plot ---
    dates = [dt.date() for dt in times]
    if len(set(dates)) == 1:
        # Single-day: seconds since midnight
        secs = [ (dt - datetime.combine(dates[0], datetime.min.time())).total_seconds() for dt in times ]
        x_time, time_format = np.array(secs), 'seconds'
    else:
        x_time, time_format = times, 'dates'

    # --- Time vector for waveform plot (ms) ---
    t_vec = np.arange(template.shape[0]) / (metadata.samples_per_second / 1000)

    return {
        'cluster_id': cluster_id,
        'tetrode': tetrode,
        'channels': channels,
        'template': template,
        'p2p': p2p,
        'abs_max': abs_max,
        'peak_sign': peak_sign,
        't': t_vec,
        'times': times,
        'x_time': x_time,
        'time_format': time_format,
        'amps': amps,
        'isi_ms': isi_ms,
        'log_isi': log_isi,
        'violations': violations,
        'vio_pct': vio_pct,
        'n_spikes': spikes.size
    }


def create_cluster_figure(data: Dict[str, Any]) -> plt.Figure:
    """
    Generate a multi-panel figure for waveform, amplitude over time, and ISI.
    """
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])

    # --- Waveform subplots ---
    chan_grid = gs[0].subgridspec(1, len(data['channels']), wspace=0.15)
    y_min, y_max = data['template'].min() * 1.1, data['template'].max() * 1.1
    for idx, ch in enumerate(data['channels']):
        ax = fig.add_subplot(chan_grid[idx])
        sns.lineplot(x=data['t'], y=data['template'][:, ch], ax=ax, linewidth=1)
        ax.axhline(0, linestyle='-', alpha=0.5)
        for tm in range(0, int(data['t'].max()) + 1, 10):
            ax.axvline(tm, linestyle='--', alpha=0.3)
        ax.set_title(f'Ch {ch}', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Amplitude (µV)')
        else:
            ax.set_yticks([])
        ax.set_xlabel('Time (ms)')
        ax.set_xlim(0, data['t'].max())
        ax.set_ylim(y_min, y_max)
        sns.despine(ax=ax)

    # --- Amplitude vs Time ---
    amp_ax = fig.add_subplot(gs[1])
    if data['time_format'] == 'seconds':
        sns.scatterplot(x=data['x_time'], y=data['amps'], ax=amp_ax, s=10, alpha=0.6)
        amp_ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}")
        )
        amp_ax.set_xlabel('Time of Day (HH:MM:SS)')
    else:
        sns.scatterplot(x=data['times'], y=data['amps'], ax=amp_ax, s=10, alpha=0.6)
        amp_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        amp_ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
        plt.setp(amp_ax.get_xticklabels(), rotation=45, ha='right')
        amp_ax.set_xlabel('Date')

    # Restrict y-axis to bulk of data
    med_amp = np.median(data['amps'])
    p5, p95 = np.percentile(data['amps'], [5, 95])
    amp_span = max(p95 - p5, 1)
    if med_amp > 0:
        lower = min(-data['abs_max'] * 0.1, med_amp - 2 * amp_span)
        upper = med_amp + 2 * amp_span
    else:
        lower = med_amp - 2 * amp_span
        upper = max(data['abs_max'] * 0.1, med_amp + 2 * amp_span)
    amp_ax.set_ylim(lower, upper)
    amp_ax.axhline(0, linestyle='-', alpha=0.3)
    amp_ax.set_ylabel('Amplitude (µV)')
    amp_ax.set_title(f"Spike Amplitude Over Time (n={data['n_spikes']})", pad=12)

    # --- ISI Distribution ---
    isi_ax = fig.add_subplot(gs[2])
    if data['log_isi'] is not None and data['log_isi'].size > 0:
        try:
            sns.kdeplot(x=data['log_isi'], fill=True, ax=isi_ax)
        except:
            sns.histplot(data['log_isi'], bins=50, stat='density', ax=isi_ax)
        for v in [0, 1, 2, 3]:
            isi_ax.axvline(v, linestyle='--', alpha=0.5)
        if data['violations'] > 0:
            isi_ax.text(
                0.05, 0.95,
                f"Refractory violations: {data['violations']} ({data['vio_pct']:.2f}%)",
                transform=isi_ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
        isi_ax.set_xlabel('ISI (log10(ms))')
        isi_ax.set_ylabel('Density')
    else:
        isi_ax.text(0.5, 0.5, 'Not enough spikes to calculate ISI', ha='center', va='center')
    isi_ax.set_title('Inter-Spike Interval Distribution', pad=12)

    fig.suptitle(f"Tetrode {data['tetrode']} Waveforms for Cluster {data['cluster_id']}", fontsize=16)
    return fig


if __name__ == "__main__":
    import logging
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    from tqdm import tqdm

    # Use non-interactive backend for PDF output
    plt.switch_backend('Agg')
    sns.set_style('whitegrid')

    @click.command()
    @click.option('--results-dir', '-d', type=click.Path(exists=True), help='Path to Kilosort results')
    @click.option('--min-amplitude', '-a', type=float, default=None, help='Minimum amplitude threshold (µV)')
    @click.option('--quiet', '-q', is_flag=True, help='Suppress progress messages')
    def main(results_dir, min_amplitude, quiet):  # pylint: disable=unused-argument
        """Generate waveform plots with optional amplitude filtering"""
        # --- Interactive prompts for missing arguments ---
        if results_dir is None:
            while True:
                user_in = click.prompt('Enter Kilosort results directory', type=str)
                try:
                    path = Path(shlex.split(user_in)[0]).expanduser()
                    if path.exists():
                        results_dir = path.resolve()
                        break
                    click.echo(f"Invalid directory: {path}")
                except Exception as err:
                    click.echo(f"Error: {err}")

        if min_amplitude is None:
            min_amplitude = click.prompt('Minimum amplitude threshold (µV)', type=float, default=0.0)

        click.echo(f"Loading data from {results_dir}...")

        # --- Load metadata and cluster info ---
        try:
            from kilosort.io import load_ops
            from sampleToDateTime import load_metadata
            ops = load_ops(Path(results_dir) / 'ops.npy')
            metadata = load_metadata(results_dir, ops)
            cluster_info = pd.read_csv(results_dir / 'cluster_group.tsv', sep='\t')
        except FileNotFoundError as err:
            click.echo(f"Missing files: {err}")
            return

        # --- Determine and sort by tetrode ---
        cluster_ids = cluster_info.loc[cluster_info['KSLabel'] != 'noise', 'cluster_id'].tolist()
        try:
            templates = np.load(results_dir / 'templates.npy')
            shanks = np.load(results_dir / 'channel_shanks.npy')
            tet_assign = {}
            for cid in cluster_ids:
                amps = np.max(np.abs(templates[cid]), axis=0)
                best = int(np.argmax(amps))
                tet_assign[cid] = int(shanks[best])
        except Exception:
            tet_assign = {cid: 0 for cid in cluster_ids}

        sorted_ids = sorted(cluster_ids, key=lambda c: (tet_assign.get(c, 0), c))

        suffix = f"_min{min_amplitude}uV" if min_amplitude and min_amplitude > 0 else ''
        pdf_file = Path(results_dir) / f"cluster_waveforms{suffix}.pdf"
        click.echo(f"Saving to {pdf_file}")

        # PDF optimization settings
        pdf_options = {
            'metadata': {
                'Title': f'Kilosort Cluster Waveforms (min amp: {min_amplitude}μV)',
                'Creator': 'plotSingleWaveformWithAmplitude.py',
            }
        }

        included = 0
        with PdfPages(pdf_file, **pdf_options) as pdf:
            # First, collect all valid clusters to create bookmarks
            bookmarks = []
            all_data = []
            
            click.echo("Preparing cluster data...")
            for cid in tqdm(sorted_ids, disable=quiet, desc="Analyzing clusters"):
                # Get the data for this cluster
                cluster_data = prepare_cluster_data(cid, metadata, min_amplitude)
                # Only proceed if we have valid data
                if cluster_data is not None:
                    all_data.append(cluster_data)
                    # Create bookmark entry: Tetrode X - Cluster Y
                    bookmarks.append(f"Tetrode {cluster_data['tetrode']} - Cluster {cid}")
            
            # Now generate figures and save to PDF
            click.echo(f"Generating PDF with {len(all_data)} clusters...")
            for i, (bookmark, cluster_data) in enumerate(zip(bookmarks, all_data), 1):
                fig = create_cluster_figure(cluster_data)
                
                # Use simplified suptitle to reduce complexity
                plt.suptitle(bookmark, fontsize=16)
                
                # Add the figure to PDF with bookmark
                pdf.savefig(fig, dpi=100, bbox_inches='tight')
                
                # Add PDF outline/bookmark
                pdf._fig_names.append(bookmark)
                
                plt.close(fig)
                
                # Update progress
                if i % 10 == 0 or i == len(all_data):
                    click.echo(f"Progress: {i}/{len(all_data)} clusters")
            
            included = len(all_data)
            
        click.echo(f"Done: {included}/{len(sorted_ids)} clusters included")

    main()
