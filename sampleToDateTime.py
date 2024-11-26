import json
from datetime import datetime, timedelta
from types import SimpleNamespace
import numpy as np
import os
from bisect import bisect_right
from metadata import Metadata
from pathlib import Path

from datetime import datetime
from pathlib import Path
from typing import List
import json

def load_metadata(dirpath: Path, ops: dict, samples_per_second: int = 30000) -> Metadata:
    """
    Loads and preprocesses metadata into a Metadata structure.

    Args:
        dirpath (Path): Path to the directory of results.
        ops (dict): Options struct used to load the channel map
        samples_per_second (int): Sampling rate (samples per second).

    Returns:
        Metadata: A structured object containing sorted metadata, sampling rate, and unique dates.
    """

    metadata_path = dirpath / "../recording_metadata.json"
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    # Ensure metadata is sorted by start_sample
    metadata_sorted = sorted(data, key=lambda x: x["start_sample"])

    # Extract start_samples for bisect
    start_samples = [entry["start_sample"] for entry in metadata_sorted]

    # Extract unique dates
    unique_dates = sorted({datetime.fromisoformat(entry["datetime"]).date() for entry in metadata_sorted})

    # Validate metadata entries
    for entry in metadata_sorted:
        if "start_sample" not in entry or "end_sample" not in entry or "datetime" not in entry:
            raise ValueError("Each metadata entry must contain 'start_sample', 'end_sample', and 'datetime'.")
        try:
            datetime.fromisoformat(entry["datetime"])  # Validate datetime format
        except ValueError as e:
            raise ValueError(f"Invalid datetime format in metadata: {entry['datetime']}") from e

    # Return Metadata object
    return Metadata(
        start_samples=start_samples,
        sorted_entries=metadata_sorted,
        samples_per_second=samples_per_second,
        unique_dates=unique_dates,
        chanMap=ops['chanMap'],
        results_dir=dirpath
    )



def sample_to_datetime(sample: int, metadata: Metadata) -> datetime:
    """
    Converts a sample number to its corresponding datetime using Metadata.

    Args:
        sample (int): The sample number to convert.
        metadata (Metadata): Metadata object containing range and timing information.

    Returns:
        datetime: The absolute datetime corresponding to the sample.

    Raises:
        ValueError: If the sample is out of range.
    """

    index = bisect_right(metadata.start_samples, sample) - 1
    if index < 0 or index >= len(metadata.sorted_entries):
        raise ValueError(f"Sample {sample} is out of range.")

    entry = metadata.sorted_entries[index]

    # Ensure the sample is within this range
    if sample > entry["end_sample"]:
        raise ValueError(f"Sample {sample} is out of range in metadata.")

    # Calculate the offset in seconds
    offset_samples = sample - entry["start_sample"]
    offset_seconds = offset_samples / metadata.samples_per_second

    # Compute and return the absolute datetime
    start_datetime = datetime.fromisoformat(entry["datetime"])
    return start_datetime + timedelta(seconds=offset_seconds)


def datetime_to_sample(query_datetime: datetime, metadata: Metadata) -> int:
    """
    Converts a datetime to its corresponding sample number using Metadata.

    Args:
        query_datetime (datetime): The datetime to convert.
        metadata (Metadata): Metadata object containing range and timing information.

    Returns:
        int: The sample number corresponding to the datetime.

    Raises:
        ValueError: If the datetime is out of range.
    """
    # Iterate through metadata to find the appropriate range
    for i, entry in enumerate(metadata.sorted_entries):
        start_datetime = datetime.fromisoformat(entry["datetime"])

        if start_datetime <= query_datetime:
            # Calculate the offset in seconds
            offset_seconds = (query_datetime - start_datetime).total_seconds()
            offset_samples = int(offset_seconds * metadata.samples_per_second)

            # Compute the sample number
            start_sample = entry["start_sample"]
            sample = start_sample + offset_samples

            # Ensure the sample is within the range
            if sample <= entry["end_sample"]:
                return sample

    raise ValueError(f"Datetime {query_datetime} is out of range.")


def group_spikes_by_time(spike_samples: np.ndarray, metadata: Metadata, th_time: float) -> (np.ndarray, np.ndarray):
    """
    Groups spikes into clusters based on inter-spike interval thresholds using Metadata.

    Args:
        spike_samples (np.ndarray): Array of spike sample indices.
        metadata (Metadata): Metadata object containing range and timing information.
        th_time (float): Time threshold for grouping spikes (in seconds).

    Returns:
        tuple: (groups, unique_groups)
            - groups: Array of group indices for each spike.
            - unique_groups: Array of unique group identifiers.
    """
    # Convert spike samples to absolute datetimes
    spike_datetimes = np.array([
        sample_to_datetime(sample, metadata) for sample in spike_samples
    ])

    # Compute time differences between consecutive spikes
    time_differences = np.array([
        (spike_datetimes[i] - spike_datetimes[i - 1]).total_seconds()
        for i in range(1, len(spike_datetimes))
    ])

    # Group spikes based on the inter-spike interval
    groups = [0]
    for time_diff in time_differences:
        if time_diff > th_time:
            groups.append(groups[-1] + 1)  # New group
        else:
            groups.append(groups[-1])  # Same group

    groups = np.array(groups)
    unique_groups = np.unique(groups)

    return groups, unique_groups