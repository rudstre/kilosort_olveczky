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
import logging

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

    metadata_path = dirpath / "recording_metadata.json"
    logging.info(f"Loading metadata from {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        data_json = json.load(f)

    # Extract file data from the new format
    if isinstance(data_json, dict) and "files" in data_json:
        data = data_json["files"]
        logging.info(f"Using 'files' key from metadata (new format)")
    else:
        # Fallback for old format
        data = data_json
        logging.info(f"Using direct metadata (old format)")

    # Ensure metadata is sorted by start_sample
    metadata_sorted = sorted(data, key=lambda x: x["start_sample"])
    
    # Log first and last entries for debugging
    if metadata_sorted:
        first = metadata_sorted[0]
        last = metadata_sorted[-1]
        logging.info(f"First entry: sample {first['start_sample']}-{first['end_sample']}, datetime {first['datetime']}")
        logging.info(f"Last entry: sample {last['start_sample']}-{last['end_sample']}, datetime {last['datetime']}")

    # Extract start_samples for bisect
    start_samples = [entry["start_sample"] for entry in metadata_sorted]

    # Extract unique dates
    unique_dates = sorted({datetime.fromisoformat(entry["datetime"]).date() for entry in metadata_sorted})
    logging.info(f"Found {len(unique_dates)} unique dates in metadata: {unique_dates}")

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



def sample_to_datetime(sample, metadata: Metadata) -> datetime:
    """
    Converts a sample number to its corresponding datetime using Metadata.

    Args:
        sample: The sample number to convert. Can be an int or a numpy array with a single value.
        metadata (Metadata): Metadata object containing range and timing information.

    Returns:
        datetime: The absolute datetime corresponding to the sample.

    Raises:
        ValueError: If the sample is out of range.
    """
    # Handle numpy array inputs by extracting the scalar value
    if hasattr(sample, 'size') and sample.size == 1:
        sample = sample.item()  # Convert numpy scalar to Python scalar
    
    # Convert start_samples to a list if it's a NumPy array
    start_samples = metadata.start_samples
    if hasattr(start_samples, 'tolist'):
        start_samples = start_samples.tolist()

    # Get index of correct metadata entry
    index = bisect_right(start_samples, sample) - 1
    if index < 0 or index >= len(metadata.sorted_entries):
        logging.error(f"Sample {sample} is out of range. Range is {min(start_samples)} to {max(start_samples) if start_samples else 'empty'}")
        # If out of range but close to the start, use the first entry
        if sample < min(start_samples) and len(metadata.sorted_entries) > 0:
            logging.warning(f"Sample {sample} is before start, using first entry")
            index = 0
        else:
            raise ValueError(f"Sample {sample} is out of range.")

    entry = metadata.sorted_entries[index]

    # Ensure the sample is within this range
    if sample > entry["end_sample"]:
        logging.warning(f"Sample {sample} is beyond end_sample {entry['end_sample']} of entry {index}, but within start_samples range")
        # If there's a next entry, use that instead
        if index + 1 < len(metadata.sorted_entries):
            logging.info(f"Using next entry (index {index+1})")
            index += 1
            entry = metadata.sorted_entries[index]
        else:
            raise ValueError(f"Sample {sample} is out of range in metadata.")

    # Calculate the offset in seconds
    offset_samples = sample - entry["start_sample"]
    offset_seconds = offset_samples / metadata.samples_per_second
    
    # Log some debug info occasionally (for every 1000th sample)
    if isinstance(sample, (int, float)) and sample % 1000 == 0:
        logging.debug(f"Sample {sample}: Using entry {index}, offset {offset_seconds:.2f}s")
    
    # Fix for potentially malformed dates in the future
    dt_str = entry["datetime"]
    dt = datetime.fromisoformat(dt_str)
    
    # If the date is way in the future (like 2025 when it's 2023), it might be a typo
    current_year = datetime.now().year
    if dt.year > current_year + 1:
        logging.warning(f"Date {dt_str} has year {dt.year} which is far in the future. Adjusting to current year {current_year}.")
        # Adjust the year to the current year
        dt = dt.replace(year=current_year)

    # Compute and return the absolute datetime
    return dt + timedelta(seconds=offset_seconds)


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
    # Ensure sorted_entries is properly converted if it's a NumPy array
    sorted_entries = metadata.sorted_entries
    if hasattr(sorted_entries, 'tolist'):
        sorted_entries = sorted_entries.tolist()
        
    # Iterate through metadata to find the appropriate range
    for i, entry in enumerate(sorted_entries):
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
    # Ensure spike_samples is properly sorted
    sorted_indices = np.argsort(spike_samples)
    sorted_samples = spike_samples[sorted_indices]
    
    # Convert spike samples to absolute datetimes
    # Use vectorize to handle arrays of samples
    vec_sample_to_datetime = np.vectorize(lambda s: sample_to_datetime(s, metadata))
    spike_datetimes = vec_sample_to_datetime(sorted_samples)

    # Compute time differences between consecutive spikes
    time_differences = np.array([
        (spike_datetimes[i] - spike_datetimes[i - 1]).total_seconds()
        for i in range(1, len(spike_datetimes))
    ])

    # Group spikes based on the inter-spike interval
    sorted_groups = [0]
    for time_diff in time_differences:
        if time_diff > th_time:
            sorted_groups.append(sorted_groups[-1] + 1)  # New group
        else:
            sorted_groups.append(sorted_groups[-1])  # Same group

    # Reorder groups to match original spike order
    groups = np.zeros_like(sorted_groups)
    groups[sorted_indices] = sorted_groups
    
    unique_groups = np.unique(groups)

    return groups, unique_groups