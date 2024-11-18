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
import spikeinterface as si
import spikeinterface.extractors as se

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set multiprocessing start method dynamically
if os.name == 'nt':  # Windows
    multiprocessing.set_start_method('spawn', force=True)
else:  # UNIX-like (Linux, macOS)
    multiprocessing.set_start_method('fork', force=True)


def build_regex_from_format(datetime_format):
    """Convert a datetime format string into a regex pattern."""
    format_to_regex = {
        "%Y": r"\d{4}",
        "%y": r"\d{2}",
        "%m": r"\d{2}",
        "%d": r"\d{2}",
        "%H": r"\d{2}",
        "%M": r"\d{2}",
        "%S": r"\d{2}",
        "_": r"_",
    }
    regex_pattern = re.escape(datetime_format)
    for directive, regex in format_to_regex.items():
        regex_pattern = regex_pattern.replace(re.escape(directive), regex)
    return regex_pattern


def extract_datetime(name, datetime_format):
    """Extract a datetime substring from a name using the provided format."""
    regex_pattern = build_regex_from_format(datetime_format)
    match = re.search(regex_pattern, name)
    if not match:
        raise ValueError(f"Could not find a datetime matching format '{datetime_format}' in '{name}'.")
    return match.group()


def prompt_user_for_datetime_format(example_name, context):
    """Prompts the user to enter the datetime format based on an example name."""
    print(
        f"""
Please provide the datetime format for the {context}.
Example {context}: {example_name}

Use Python datetime format codes:
- %Y: Year (4 digits), %y: Year (2 digits)
- %m: Month (2 digits), %d: Day (2 digits)
- %H: Hour (24-hour clock), %M: Minute, %S: Second
- _: Literal underscore

For example, if the {context} contains '240830_145124', enter: '%y%m%d_%H%M%S'.
"""
    )
    while True:
        datetime_format = input("Enter the datetime format: ").strip()
        try:
            regex_pattern = build_regex_from_format(datetime_format)
            if re.search(regex_pattern, example_name):
                return datetime_format
            else:
                raise ValueError
        except ValueError:
            logging.error(f"The format '{datetime_format}' does not match the example '{example_name}'. Please try again.")


def get_subfolders(folder_path):
    """Retrieves all subfolders in the given folder."""
    return [os.path.join(folder_path, subfolder) for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]


def validate_output_folder(output_folder):
    """Validates the output folder, deleting it if the user agrees."""
    if os.path.exists(output_folder):
        response = input(f"Output folder '{output_folder}' already exists. Delete it? (yes/no): ").strip().lower()
        if response == "yes":
            shutil.rmtree(output_folder)
            logging.info(f"Deleted existing output folder: {output_folder}")
        else:
            logging.info("Operation cancelled by user.")
            exit()


def sort_files_by_datetime(files, datetime_format):
    """Sorts a list of files by datetime extracted from their names."""
    return sorted(files, key=lambda x: datetime.strptime(extract_datetime(os.path.basename(x), datetime_format), datetime_format))


def save_metadata(metadata, output_folder):
    """Saves metadata to a JSON file."""
    metadata_path = os.path.join(output_folder, "recording_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")


def main(input_folder, output_folder, n_jobs):
    # Get recording folders
    recording_folders = get_subfolders(input_folder)
    if not recording_folders:
        logging.error("No recording folders found in the specified input folder.")
        exit()

    # Ask the user for folder datetime format
    folder_example = os.path.basename(recording_folders[0])
    folder_datetime_format = prompt_user_for_datetime_format(folder_example, "folder name")

    # Sort recording folders by datetime
    recording_folders = sort_files_by_datetime(recording_folders, folder_datetime_format)

    # Find the first file for datetime format prompt
    file_example = None
    for folder in recording_folders:
        rhd_files = glob(os.path.join(folder, "*.rhd"))
        if rhd_files:
            file_example = os.path.basename(rhd_files[0])
            break
    if not file_example:
        logging.error("No .rhd files found in any of the folders.")
        exit()

    # Ask the user for file datetime format
    file_datetime_format = prompt_user_for_datetime_format(file_example, "file name")

    # Initialize data
    recordings = []
    metadata = []
    cumulative_samples = 0

    for folder in recording_folders:
        try:
            folder_datetime_str = extract_datetime(os.path.basename(folder), folder_datetime_format)
            folder_datetime = datetime.strptime(folder_datetime_str, folder_datetime_format)
            human_readable_datetime = folder_datetime.strftime("%B %d, %Y, %H:%M")
        except ValueError:
            human_readable_datetime = "an unknown date and time"

        rhd_files = glob(os.path.join(folder, "*.rhd"))
        if not rhd_files:
            logging.warning(
                f"No .rhd files found in the folder from {human_readable_datetime}: {folder}. Skipping."
            )
            continue

        logging.info(f"Processing recordings from {human_readable_datetime}: {folder}")

        sorted_files = sort_files_by_datetime(rhd_files, file_datetime_format)

        for recording_path in tqdm(sorted_files, desc=f"Processing recordings from {human_readable_datetime}"):
            recording = se.read_intan(recording_path, stream_id="0")
            recordings.append(recording)

            # Extract datetime from the file name
            file_datetime_str = extract_datetime(os.path.basename(recording_path), file_datetime_format)
            file_datetime = datetime.strptime(file_datetime_str, file_datetime_format)

            # Update metadata
            num_samples = recording.get_num_frames()
            metadata.append({
                "file_name": os.path.basename(recording_path),
                "start_sample": cumulative_samples,
                "end_sample": cumulative_samples + num_samples - 1,
                "datetime": file_datetime.isoformat(),  # Save in ISO 8601 format
            })
            cumulative_samples += num_samples

    # Concatenate recordings
    if len(recordings) > 1:
        logging.info("Concatenating all recordings into one.")
        concatenated_recording = si.concatenate_recordings(recordings)
    elif recordings:
        concatenated_recording = recordings[0]
    else:
        logging.error("No valid recordings found.")
        exit()

    # Validate output folder and save concatenated recording
    # validate_output_folder(output_folder)
    logging.info("Saving concatenated recording...")
    # concatenated_recording.save(folder=output_folder, n_jobs=n_jobs)

    # Save metadata
    save_metadata(metadata, output_folder)

    logging.info("Process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate .rhd recordings from multiple folders into a single file.")
    parser.add_argument("-i", "--input", help="Input folder containing recording folders.")
    parser.add_argument("-o", "--output", help="Output folder for the concatenated recording.")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of jobs to use (default: all available cores).")

    args = parser.parse_args()

    # Explicitly ask for input/output folders if not provided
    if not args.input:
        args.input = input("Please enter the input folder containing recording folders: ").strip()
    if not os.path.isdir(args.input):
        logging.error(f"The input folder '{args.input}' does not exist.")
        exit()

    if not args.output:
        args.output = input("Please enter the output folder for the concatenated recording: ").strip()

    main(args.input, args.output, args.jobs)
    multiprocessing.active_children()  # Ensure cleanup of subprocesses
