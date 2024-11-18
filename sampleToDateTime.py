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


def sample_to_datetime(sample, metadata, sample_rate=30000):
    """Convert a sample number to the corresponding datetime.

    Args:
        sample (int): The sample number in the concatenated recording.
        metadata (any): JSON metadata file.
        sample_rate (int): The sample rate of the recording in Hz (default is 30kHz).

    Returns:
        datetime: The corresponding datetime for the given sample.
    """

    # Find the correct recording segment
    for segment in metadata:
        start_sample = segment['start_sample']
        end_sample = segment['end_sample']

        if start_sample <= sample <= end_sample:
            # Calculate the offset in time from the start of the segment
            offset_samples = sample - start_sample
            offset_seconds = offset_samples / sample_rate
            offset_timedelta = timedelta(seconds=offset_seconds)

            # Calculate the corresponding datetime
            start_time = date_parser.parse(segment['datetime'])
            return start_time + offset_timedelta

    # If the sample is not found in any segment
    raise ValueError(f"Sample {sample} is out of bounds for the concatenated recording.")
