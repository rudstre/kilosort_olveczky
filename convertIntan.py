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


def load_recording_paths(input_folder, recursive):
    """Retrieve subfolders or .rhd files directly depending on the recursive flag."""
    if recursive:
        subfolders = [os.path.join(input_folder, sf) for sf in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, sf))]
        if not subfolders:
            raise ValueError("No subfolders found in the input folder.")
        return subfolders, True
    rhd_files = glob(os.path.join(input_folder, "*.rhd"))
    if not rhd_files:
        raise ValueError("No .rhd files found in the input folder.")
    return rhd_files, False


def process_subfolder(args):
    """Process a single subfolder."""
    folder, file_datetime_formats = args
    rhd_files = glob(os.path.join(folder, "*.rhd"))
    if not rhd_files:
        logging.warning(f"No .rhd files found in {folder}. Skipping.")
        return [], [], 0

    # Sort recordings within the subfolder by datetime
    for fmt in file_datetime_formats:
        try:
            rhd_files = sorted(
                rhd_files,
                key=lambda p: datetime.strptime(
                    extract_datetime(os.path.basename(p), fmt),
                    fmt
                )
            )
            break
        except ValueError:
            continue

    recordings, metadata, cumulative_samples = [], [], 0

    for recording_path in rhd_files:
        recording = se.read_intan(recording_path, stream_id="0", ignore_integrity_checks=True)
        recordings.append(recording)

        # Attempt to parse datetime for each recording
        file_datetime = None
        for fmt in file_datetime_formats:
            try:
                file_datetime_str = extract_datetime(os.path.basename(recording_path), fmt)
                file_datetime = datetime.strptime(file_datetime_str, fmt)
                break
            except ValueError:
                continue

        # If parsing fails, prompt user for new format
        while not file_datetime:
            new_format = prompt_user_for_datetime_format(os.path.basename(recording_path), "file name")
            file_datetime_formats.append(new_format)
            try:
                file_datetime_str = extract_datetime(os.path.basename(recording_path), new_format)
                file_datetime = datetime.strptime(file_datetime_str, new_format)
            except ValueError:
                continue

        # Update metadata
        num_samples = recording.get_num_frames()
        metadata.append({
            "file_name": os.path.basename(recording_path),
            "start_sample": cumulative_samples,
            "end_sample": cumulative_samples + num_samples - 1,
            "datetime": file_datetime.isoformat(),
        })
        cumulative_samples += num_samples

    return recordings, metadata, cumulative_samples


def main(input_folder, output_folder, n_jobs, recursive):
    recording_paths, is_recursive = load_recording_paths(input_folder, recursive)

    # Handle recursive mode: Parse and sort subfolders by datetime
    if is_recursive:
        folder_example = os.path.basename(recording_paths[0])
        folder_datetime_format = prompt_user_for_datetime_format(folder_example, "folder name")
        try:
            recording_paths = sorted(
                recording_paths,
                key=lambda p: datetime.strptime(
                    extract_datetime(os.path.basename(p), folder_datetime_format),
                    folder_datetime_format
                )
            )
        except ValueError as e:
            logging.error(f"Error sorting subfolders: {e}")
            exit()
    else:
        # Non-recursive mode: Treat input folder as a single "folder"
        recording_paths = [input_folder]

    # Example file for datetime parsing
    file_example = next((os.path.basename(f) for p in recording_paths for f in glob(os.path.join(p, "*.rhd"))), None)
    if not file_example:
        logging.error("No .rhd files found.")
        exit()

    # Prompt for datetime format for file names
    file_datetime_formats = [prompt_user_for_datetime_format(file_example, "file name")]

    # Validate output folder
    validate_output_folder(output_folder)

    # Initialize progress bar
    with tqdm(total=len(recording_paths), desc="Processing Subfolders", position=0, leave=True) as progress_bar:
        args = [(p, file_datetime_formats) for p in recording_paths]

        # Start multiprocessing pool
        with Pool(n_jobs) as pool:
            results = pool.map(process_subfolder, args)

            # Update progress bar as each subfolder is completed
            for _ in results:
                progress_bar.update(1)

    # Combine results
    all_recordings, all_metadata, total_samples = [], [], 0
    for recs, meta, samples in results:
        all_recordings.extend(recs)
        all_metadata.extend(meta)
        total_samples += samples

    # Concatenate recordings
    if len(all_recordings) > 1:
        concatenated_recording = si.concatenate_recordings(all_recordings)
    elif all_recordings:
        concatenated_recording = all_recordings[0]
    else:
        logging.error("No valid recordings found.")
        exit()

    # Save output
    concatenated_recording.save(dtype="int16", format="binary", folder=output_folder, n_jobs=n_jobs)
    save_metadata(all_metadata, output_folder)
    logging.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate .rhd recordings into a single file.")
    parser.add_argument("-i", "--input", help="Input folder containing recordings or recording folders.")
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
