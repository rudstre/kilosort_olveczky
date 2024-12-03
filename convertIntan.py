import multiprocessing
import os
import re
import json
import shutil
import logging
import argparse
from glob import glob
from datetime import datetime
from itertools import islice
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import spikeinterface as si
import spikeinterface.extractors as se
import psutil
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_process_memory_usage():
    """Get memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024**2)  # Convert to MB
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / (1024**2)
    return mem_usage


# Initialize previous counters
prev_read_bytes = 0
prev_write_bytes = 0
def log_system_status():
    global prev_read_bytes, prev_write_bytes
    process = psutil.Process(os.getpid())
    memory = get_process_memory_usage()
    io_counters = process.io_counters()

    # Calculate incremental I/O
    read_bytes = io_counters.read_bytes - prev_read_bytes
    write_bytes = io_counters.write_bytes - prev_write_bytes

    # Update previous counters
    prev_read_bytes = io_counters.read_bytes
    prev_write_bytes = io_counters.write_bytes

    logging.info(f"Memory Usage: {memory.percent}%")
    logging.info(f"Process Disk Read since last check: {read_bytes / (1024**2):.2f} MB, "
                 f"Process Disk Write since last check: {write_bytes / (1024**2):.2f} MB")

def build_regex_from_format(datetime_format):
    """Convert a datetime format string into a regex pattern."""
    format_to_regex = {
        "%Y": r"\d{4}", "%y": r"\d{2}", "%m": r"\d{2}", "%d": r"\d{2}",
        "%H": r"\d{2}", "%M": r"\d{2}", "%S": r"\d{2}",
    }
    regex_pattern = ""
    i = 0
    while i < len(datetime_format):
        if datetime_format[i:i+2] in format_to_regex:
            regex_pattern += format_to_regex[datetime_format[i:i+2]]
            i += 2
        else:
            regex_pattern += re.escape(datetime_format[i])
            i += 1
    return regex_pattern


def extract_datetime(name, datetime_format):
    """Extract a datetime substring from a name using the provided format."""
    regex_pattern = build_regex_from_format(datetime_format)
    match = re.search(regex_pattern, name)
    if not match:
        raise ValueError(f"No datetime matching format '{datetime_format}' found in '{name}'.")
    return match.group()


def prompt_user_for_datetime_format(example_name, context):
    """Prompt the user to enter the datetime format based on an example name."""
    print(
        f"\nProvide the datetime format for the {context}.\n"
        f"Example {context}: {example_name}\n"
        "Use Python datetime codes (e.g., '%y%m%d_%H%M%S' for '240830_145124')."
    )
    while True:
        datetime_format = input("Enter the datetime format: ").strip()
        try:
            if re.search(build_regex_from_format(datetime_format), example_name):
                return datetime_format
        except ValueError:
            pass
        logging.error(f"Invalid format '{datetime_format}'. Try again.")


def validate_output_folder(output_folder):
    """Validate the output folder, deleting it if the user agrees."""
    if os.path.exists(output_folder):
        if input(f"Output folder '{output_folder}' exists. Delete it? (yes/no): ").strip().lower() == "yes":
            shutil.rmtree(output_folder)
            logging.info(f"Deleted existing output folder: {output_folder}")
        else:
            logging.info("Operation cancelled by user.")
            exit()


def save_metadata(metadata, output_folder):
    """Save metadata to a JSON file."""
    metadata_path = os.path.join(output_folder, "recording_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")


def load_rhd_files(input_folder, recursive):
    """Retrieve all .rhd files in the folder or subfolders based on the recursive flag."""
    if recursive:
        rhd_files = glob(os.path.join(input_folder, "**", "*.rhd"), recursive=True)
    else:
        rhd_files = glob(os.path.join(input_folder, "*.rhd"))
    if not rhd_files:
        raise ValueError("No .rhd files found.")
    return rhd_files


def process_rhd_file(args):
    """Process a single .rhd file."""
    file_path, file_datetime_formats = args
    try:
        recording = se.read_intan(file_path, stream_id="0", ignore_integrity_checks=True)

        # Extract datetime from the file name
        file_datetime = None
        for fmt in file_datetime_formats:
            try:
                file_datetime_str = extract_datetime(os.path.basename(file_path), fmt)
                file_datetime = datetime.strptime(file_datetime_str, fmt)
                break
            except ValueError:
                continue

        # If parsing fails, prompt user for new format
        while not file_datetime:
            new_format = prompt_user_for_datetime_format(os.path.basename(file_path), "file name")
            file_datetime_formats.append(new_format)
            try:
                file_datetime_str = extract_datetime(os.path.basename(file_path), new_format)
                file_datetime = datetime.strptime(file_datetime_str, new_format)
            except ValueError:
                continue

        # Prepare metadata
        num_samples = recording.get_num_frames()
        metadata = {
            "file_name": os.path.basename(file_path),
            "start_sample": 0,
            "end_sample": num_samples - 1,
            "datetime": file_datetime.isoformat(),
        }
        return recording, metadata

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None, None


def chunked_iterable(iterable, size):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def process_and_concatenate_batches(rhd_files, file_datetime_formats, batch_size, output_folder, n_jobs):
    """Process .rhd files in batches, concatenate them, and manage memory."""
    all_metadata = []  # To collect metadata
    concatenated_batches = []  # To store intermediate concatenated recordings

    # Group files into smaller batches
    file_batches = list(chunked_iterable(rhd_files, batch_size))

    for batch_idx, file_batch in enumerate(file_batches):
        logging.info(f"Processing batch {batch_idx + 1}/{len(file_batches)} with {len(file_batch)} files.")

        # Process the current batch
        batch_results = []
        for file_path in file_batch:
            result = process_rhd_file((file_path, file_datetime_formats))
            log_system_status()
            if result:
                batch_results.append(result)

        # Collect recordings and metadata
        batch_recordings = []
        for recording, metadata in batch_results:
            if recording and metadata:
                batch_recordings.append(recording)
                all_metadata.append(metadata)

        # Concatenate recordings in the batch
        if len(batch_recordings) > 1:
            concatenated_batch = si.concatenate_recordings(batch_recordings)
        elif batch_recordings:
            concatenated_batch = batch_recordings[0]
        else:
            logging.error(f"No valid recordings found in batch {batch_idx + 1}. Skipping.")
            continue

        # Store the concatenated batch
        concatenated_batches.append(concatenated_batch)

        # Explicitly delete intermediate objects to free memory
        del batch_recordings
        del batch_results
        log_system_status()

    # Final concatenation of all batches
    if len(concatenated_batches) > 1:
        final_concatenated = si.concatenate_recordings(concatenated_batches)
    elif concatenated_batches:
        final_concatenated = concatenated_batches[0]
    else:
        logging.error("No valid recordings found after processing all batches.")
        return None, all_metadata

    # Save the final concatenated recording
    final_concatenated.save(dtype="int16", format="binary", folder=output_folder, n_jobs=n_jobs)
    return final_concatenated, all_metadata


def main(input_folder, output_folder, n_jobs, recursive):
    # Load all .rhd files
    rhd_files = load_rhd_files(input_folder, recursive)

    # Example file for datetime format prompting
    file_example = os.path.basename(rhd_files[0])
    file_datetime_formats = [prompt_user_for_datetime_format(file_example, "file name")]

    # Validate output folder
    validate_output_folder(output_folder)

    # Process and concatenate recordings in batches
    batch_size = 50  # Define batch size
    final_recording, all_metadata = process_and_concatenate_batches(
        rhd_files, file_datetime_formats, batch_size, output_folder, n_jobs
    )

    if final_recording:
        save_metadata(all_metadata, output_folder)
        logging.info("Process completed successfully.")
    else:
        logging.error("Processing failed. No valid output was produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate .rhd recordings into a single file.")
    parser.add_argument("-i", "--input", help="Input folder containing recordings.")
    parser.add_argument("-o", "--output", help="Output folder for the concatenated recording.")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of jobs (default: all cores).")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search for recordings in subfolders.")

    args = parser.parse_args()
    args.input = args.input or input("Enter the path to the input folder: ").strip()
    while not os.path.isdir(args.input):
        args.input = input("Invalid input folder. Try again: ").strip()

    args.output = args.output or input("Enter the path for the output folder: ").strip()
    args.recursive = args.recursive or input("Search for recordings in subfolders? (yes/no): ").strip().lower() in ("yes", "y")
    main(args.input, args.output, args.jobs, args.recursive)
