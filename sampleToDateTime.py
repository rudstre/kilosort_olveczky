import json
from datetime import timedelta
from dateutil import parser as date_parser
import os


def load_metadata(metadata_path):
    """Load metadata from a JSON file."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file '{metadata_path}' not found.")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


import numpy as np
from datetime import timedelta
from dateutil import parser as date_parser

def samples_to_relative_seconds(samples, metadata, sample_rate=30000):
    """
    Convert an array of sample numbers to seconds relative to the start of the first segment.

    Args:
        samples (array-like): Array of sample numbers in the concatenated recording.
        metadata (list): List of metadata dictionaries for each segment.
        sample_rate (int): The sample rate of the recording in Hz (default is 30kHz).

    Returns:
        np.ndarray: Array of seconds relative to the start of the first segment.
    """
    samples = np.asarray(samples)  # Ensure samples is a NumPy array
    relative_seconds = np.empty(samples.shape, dtype=float)  # Preallocate result array

    # Get the start datetime of the first segment
    first_segment_start_time = date_parser.parse(metadata[0]['datetime']).timestamp()
    first_segment_start_sample = metadata[0]['start_sample']

    # Iterate over metadata segments
    for segment in metadata:
        start_sample = segment['start_sample']
        end_sample = segment['end_sample']
        segment_start_time = date_parser.parse(segment['datetime']).timestamp()

        # Find samples within the current segment
        mask = (samples >= start_sample) & (samples <= end_sample)
        segment_samples = samples[mask]

        # Compute relative seconds
        relative_start_time = segment_start_time - first_segment_start_time
        offset_samples = segment_samples - start_sample
        relative_seconds[mask] = relative_start_time + offset_samples / sample_rate

    return relative_seconds


