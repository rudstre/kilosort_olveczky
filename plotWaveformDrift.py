from pathlib import PosixPath

import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams

# outputs saved to results_dir

root = tk.Tk()
root.withdraw()  # Hide the root window

results_dir = filedialog.askdirectory()

if results_dir:
    print("Selected folder:", results_dir)
    results_dir = PosixPath(results_dir)  # Explicitly use PosixPath

# ops = load_ops(results_dir / 'ops.npy')
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
chan_map = np.load(results_dir / 'channel_map.npy')
templates = np.load(results_dir / 'templates.npy')
chan_best = (templates ** 2).sum(axis=1).argmax(axis=-1)
chan_best = chan_map[chan_best]
amplitudes = np.load(results_dir / 'amplitudes.npy')
st = np.load(results_dir / 'spike_times.npy')
clu = np.load(results_dir / 'spike_clusters.npy')
firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
# dshift = ops['dshift']

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
gray = .5 * np.ones(3)

fig = plt.figure(figsize=(10, 10), dpi=100)
grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

ax = fig.add_subplot(grid[1, 0])
nb = ax.hist(firing_rates, 20, color=gray)
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1, 1])
nb = ax.hist(camps, 20, color=gray)
ax.set_xlabel('amplitude')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1, 2])
nb = ax.hist(np.minimum(100, contam_pct), np.arange(0, 105, 5), color=gray)
ax.plot([10, 10], [0, nb[0].max()], 'k--')
ax.set_xlabel('% contamination')
ax.set_ylabel('# of units')
ax.set_title('< 10% = good units')

for k in range(2):
    ax = fig.add_subplot(grid[2, k])
    is_ref = contam_pct < 10.
    ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
    ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xlabel('firing rate (Hz)')
    ax.legend()
    if k == 1:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('loglog')

plt.show()
