import os
import re
import json
import shutil
import logging
import argparse
from glob import glob
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import spikeinterface as si
import spikeinterface.extractors as se

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set multiprocessing start method dynamically
multiprocessing.set_start_method('spawn' if os.name == 'nt' else 'fork', force=True)


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


def main(input_folder, output_folder, n_jobs, recursive):
    # Load all .rhd files
    rhd_files = load_rhd_files(input_folder, recursive)

    # Example file for datetime format prompting
    file_example = os.path.basename(rhd_files[0])
    file_datetime_formats = [prompt_user_for_datetime_format(file_example, "file name")]

    # Validate output folder
    validate_output_folder(output_folder)

    # Initialize progress bar
    with tqdm(total=len(rhd_files), desc="Processing Files", position=0, leave=True) as progress_bar:
        args = [(file, file_datetime_formats) for file in rhd_files]

        # Start multiprocessing pool
        with Pool(n_jobs) as pool:
            # Use imap_unordered for asynchronous processing
            results = []
            for result in pool.imap_unordered(process_rhd_file, args, chunksize=1):
                results.append(result)
                progress_bar.update(1)

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
    multiprocessing.active_children()
