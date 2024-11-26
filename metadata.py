from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from datetime import date

@dataclass
class Metadata:
    start_samples: List[int]  # Sorted list of start sample indices
    sorted_entries: List[Dict[str, Any]]  # Metadata entries sorted by start_sample
    samples_per_second: int  # Sampling rate (samples per second)
    unique_dates: List[date] = field(default_factory=list)

    # Additional attributes
    chanMap: Any = None  # Channel map (optional, structure depends on your data)
    results_dir: Path = None  # Path to results directory
    th_time: float = 3600  # Threshold time for grouping (in seconds)
    cmap: str = 'RdYlGn'  # Colormap for plotting
    th_spikes: int = 10  # Minimum spikes per group for filtering
