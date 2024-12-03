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

def log_system_status():
    """Log CPU, memory, and disk I/O stats."""
    memory = psutil.virtual_memory()
    io_counters = psutil.disk_io_counters()
    logging.info(f"Memory Usage: {memory.percent}%")
    logging.info(f"Disk Read: {io_counters.read_bytes / (1024**2):.2f} MB, "
                 f"Disk Write: {io_counters.write_bytes / (1024**2):.2f} MB")

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


def process_rhd_file_batch(args_batch):
    """Process a batch of .rhd files."""
    batch_results = []
    for args in args_batch:
        result = process_rhd_file(args)
        if len(batch_results) % 10 == 0:  # Log system status every 10 files
            log_system_status()
        if result:
            batch_results.append(result)
    return batch_results


def main(input_folder, output_folder, n_jobs, recursive):
    # Load all .rhd files
    rhd_files = load_rhd_files(input_folder, recursive)

    # Example file for datetime format prompting
    file_example = os.path.basename(rhd_files[0])
    file_datetime_formats = [prompt_user_for_datetime_format(file_example, "file name")]

    # Validate output folder
    validate_output_folder(output_folder)

    # Group files into smaller batches
    batch_size = 50
    file_batches = list(chunked_iterable(rhd_files, batch_size))

    # Initialize progress bar
    with tqdm(total=len(rhd_files), desc="Processing Files", position=0, leave=True) as progress_bar:
        args_batches = [
            [(file, file_datetime_formats) for file in batch] for batch in file_batches
        ]

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_rhd_file_batch, args_batch) for args_batch in args_batches]

            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                progress_bar.update(len(args_batches[0]))  # Update for batch size

    # Combine and sort results by datetime
    all_recordings, all_metadata = [], []
    for recording, metadata in results:
        if recording and metadata:
            all_recordings.append((recording, metadata["datetime"]))
            all_metadata.append(metadata)

    # Sort recordings and metadata by datetime
    all_recordings.sort(key=lambda x: x[1])
    all_metadata.sort(key=lambda x: x["datetime"])

    # Extract sorted recordings
    sorted_recordings = [rec[0] for rec in all_recordings]

    # Concatenate recordings
    if len(sorted_recordings) > 1:
        concatenated_recording = si.concatenate_recordings(sorted_recordings)
    elif sorted_recordings:
        concatenated_recording = sorted_recordings[0]
    else:
        logging.error("No valid recordings found.")
        exit()

    # Save output
    concatenated_recording.save(dtype="int16", format="binary", folder=output_folder, n_jobs=n_jobs)
    save_metadata(all_metadata, output_folder)
    logging.info("Process completed successfully.")


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
