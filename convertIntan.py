import os
import re
import json
import shlex
import shutil
import logging
import argparse
import numpy as np
from glob import glob
from datetime import datetime
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

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
    """Try default format first, then prompt the user to enter the datetime format if needed."""
    # Try the default format first (%y%m%d_%H%M%S)
    default_format = "%y%m%d_%H%M%S"
    try:
        if re.search(build_regex_from_format(default_format), example_name):
            logging.info(f"Using default datetime format '{default_format}' for {context}")
            return default_format
    except ValueError:
        pass
    
    # If default format doesn't work, prompt the user
    print(
        f"\nCould not parse {context} with default format '%y%m%d_%H%M%S'.\n"
        f"Example {context}: {example_name}\n"
        "Please provide the datetime format using Python datetime codes."
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


def reorder_channels(recording, channel_order):
    """Reorder channels in the recording based on the specified order."""
    if channel_order is None:
        return recording
        
    logging.info("Reordering channels based on provided channel order...")
    
    # Get the actual channel IDs from the recording
    all_channel_ids = recording.get_channel_ids()
    n_channels = recording.get_num_channels()
    
    # Validate the channel order indices
    if max(channel_order) >= n_channels or min(channel_order) < 0:
        raise ValueError(f"Channel indices must be in range [0, {n_channels-1}]")
    
    # Check if the channel order has the right number of channels
    if len(channel_order) < n_channels:
        # If user provided fewer indices than channels, ask if they want to proceed with only those channels
        missing_count = n_channels - len(channel_order)
        logging.warning(f"Channel order contains only {len(channel_order)} indices but recording has {n_channels} channels.")
        logging.warning(f"This will result in {missing_count} channels being dropped from the output.")
        
        if input("Continue with partial channel selection? (yes/no): ").strip().lower() != "yes":
            raise ValueError("Channel reordering cancelled by user.")
    
    # Map numeric indices to actual channel IDs
    try:
        channel_ids_to_use = [all_channel_ids[idx] for idx in channel_order if idx < n_channels]
        
        # Use SpikeInterface's SubRecordingExtractor to reorder channels using the IDs
        reordered_recording = recording.channel_slice(channel_ids_to_use)
        logging.info(f"Channels reordered successfully. Output has {len(channel_ids_to_use)} channels.")
        
        return reordered_recording
    except Exception as e:
        logging.error(f"Error during channel reordering: {str(e)}")
        raise ValueError(f"Failed to reorder channels: {str(e)}")


def main(input_folder, output_folder, n_jobs, recursive, channel_order=None, min_datetime=None, max_datetime=None, datetime_formats=None):
    # Load all .rhd files
    rhd_files = load_rhd_files(input_folder, recursive)

    # Example file for datetime format prompting
    file_example = os.path.basename(rhd_files[0])
    # Use provided formats if available, otherwise prompt
    file_datetime_formats = datetime_formats or [prompt_user_for_datetime_format(file_example, "file name")]
    
    # Pre-filter files by datetime if filters are specified
    if min_datetime is not None or max_datetime is not None:
        logging.info("Pre-filtering files by datetime range...")
        filtered_files = []
        skipped_count = 0
        
        for file_path in tqdm(rhd_files, desc="Filtering files"):
            try:
                # Extract datetime from the file name
                file_datetime = None
                for fmt in file_datetime_formats:
                    try:
                        file_datetime_str = extract_datetime(os.path.basename(file_path), fmt)
                        file_datetime = datetime.strptime(file_datetime_str, fmt)
                        break
                    except ValueError:
                        continue
                
                # Skip file if outside datetime range
                if file_datetime:
                    if (min_datetime and file_datetime < min_datetime) or (max_datetime and file_datetime > max_datetime):
                        skipped_count += 1
                        continue
                    filtered_files.append(file_path)
                else:
                    # If can't parse datetime, include the file to be safe
                    filtered_files.append(file_path)
            except Exception as e:
                logging.debug(f"Error pre-filtering file {file_path}: {e}")
                # Include file if there's an error parsing datetime
                filtered_files.append(file_path)
        
        logging.info(f"Pre-filtered {skipped_count} files outside the datetime range.")
        logging.info(f"Processing {len(filtered_files)} files that match the datetime range.")
        
        # Use filtered files for processing
        rhd_files = filtered_files

    # Validate output folder
    validate_output_folder(output_folder)
    
    if not rhd_files:
        logging.error("No valid recordings found within the specified datetime range.")
        exit()

    # Initialize progress bar
    with tqdm(total=len(rhd_files), desc="Processing Files", position=0, leave=True) as progress_bar:
        args = [(file, file_datetime_formats) for file in rhd_files]

        # Start multiprocessing pool
        results = []
        with Pool(n_jobs) as pool:
            # Define callback for progress updates
            def update_progress(result):
                results.append(result)
                progress_bar.update(1)

            # Use map_async for non-blocking processing
            async_results = [
                pool.apply_async(process_rhd_file, (arg,), callback=update_progress)
                for arg in args
            ]

            # Wait for all tasks to complete
            for async_result in async_results:
                async_result.wait()

    # Combine results
    all_recordings, all_metadata = [], []
    for recording, metadata in results:
        if recording and metadata:
            all_recordings.append((recording, metadata["datetime"]))
            all_metadata.append(metadata)

    if not all_recordings:
        logging.error("No valid recordings found.")
        exit()

    # Sort recordings and metadata by datetime
    all_recordings.sort(key=lambda x: x[1])
    all_metadata.sort(key=lambda x: x["datetime"])

    # Log the selected date range
    if min_datetime or max_datetime:
        date_range = f"from {min_datetime.isoformat() if min_datetime else 'earliest'} to {max_datetime.isoformat() if max_datetime else 'latest'}"
        logging.info(f"Concatenating {len(all_recordings)} files {date_range}")

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
        
    # Apply channel reordering if specified
    if channel_order:
        try:
            concatenated_recording = reorder_channels(concatenated_recording, channel_order)
        except ValueError as e:
            logging.error(f"Channel reordering failed: {e}")
            if input("Continue without channel reordering? (yes/no): ").strip().lower() != "yes":
                logging.info("Operation cancelled by user.")
                exit()

    # Apply filtering and referencing
    logging.info("Filtering raw data (bandpass and common reference)...")
    recording_band = sp.bandpass_filter(recording=concatenated_recording, freq_min=300, freq_max=6000)
    final_recording = sp.common_reference(recording=recording_band, operator="median")
    logging.info("Data filtered.")

    # Save channel order information in metadata if used
    processing_metadata = {
        "processing_info": {
            "datetime_range": {
                "min_datetime": min_datetime.isoformat() if min_datetime else None,
                "max_datetime": max_datetime.isoformat() if max_datetime else None
            }
        }
    }
    
    if channel_order:
        processing_metadata["processing_info"]["channel_order"] = {
            "channel_order_used": True,
            "channel_order": channel_order.tolist() if isinstance(channel_order, np.ndarray) else list(channel_order)
        }
    
    all_metadata.append(processing_metadata)

    # Save output
    final_recording.save(dtype="int16", format="binary", folder=output_folder, n_jobs=n_jobs)
    save_metadata(all_metadata, output_folder)
    logging.info("Process completed successfully.")


def parse_datetime(datetime_str, format_str=None):
    """Parse a datetime string with multiple format attempts."""
    if not datetime_str:
        return None
        
    if format_str:
        try:
            return datetime.strptime(datetime_str, format_str)
        except ValueError:
            pass
    
    # Try common formats, with our preferred default first
    formats = [
        "%y%m%d_%H%M%S",  # Our default format (e.g., 240830_145124)
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y%m%d_%H%M%S",
        "%Y%m%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
            
    raise ValueError(f"Could not parse datetime: {datetime_str}. Please provide a format like 'YYYY-MM-DD HH:MM:SS'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate .rhd recordings into a single file.")
    parser.add_argument("-i", "--input", help="Input folder containing recordings.")
    parser.add_argument("-o", "--output", help="Output folder for the concatenated recording.")
    parser.add_argument("-j", "--jobs", type=int, default=multiprocessing.cpu_count(), help="Number of jobs (default: all cores).")
    parser.add_argument("-r", "--recursive", action="store_true", help="Search for recordings in subfolders.")
    parser.add_argument("--min-datetime", help="Minimum datetime to include (format: YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--max-datetime", help="Maximum datetime to include (format: YYYY-MM-DD HH:MM:SS)")

    args = parser.parse_args()
    
    # Helper function to sanitize path inputs
    def sanitize_path(path):
        """
        Sanitizes a path string by:
        1. Removing surrounding quotes (both single and double)
        2. Expanding user directories (~/...)
        3. Resolving environment variables ($HOME/...)
        4. Converting to absolute path if needed
        """
        if not path:
            return path
            
        # Strip whitespace
        path = path.strip()
        
        try:
            # Use shlex to parse shell-like syntax, including quotes
            if path.startswith('"') or path.startswith("'") or "\\'" in path or '\\"' in path:
                try:
                    path = shlex.split(path)[0]
                except Exception:
                    # If shlex fails, try simple quote stripping
                    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
                        path = path[1:-1]
        except Exception as e:
            logging.warning(f"Error parsing path quotes: {e}")
        
        # Expand user directory (~/...)
        path = os.path.expanduser(path)
        
        # Expand environment variables ($HOME/...)
        path = os.path.expandvars(path)
        
        return path
    
    # Get and sanitize input folder path
    args.input = sanitize_path(args.input or input("Enter the path to the input folder: ").strip())
    while not os.path.isdir(args.input):
        print(f"Invalid input folder: '{args.input}'")
        args.input = sanitize_path(input("Please enter a valid path: ").strip())

    # Get and sanitize output folder path
    args.output = sanitize_path(args.output or input("Enter the path for the output folder: ").strip())
    args.recursive = args.recursive or input("Search for recordings in subfolders? (yes/no): ").strip().lower() in ("yes", "y")
    
    # Process datetime filters
    min_datetime = None
    max_datetime = None
    file_datetime_formats = []  # Store datetime formats to reuse
    
    # Analyze all available recordings to show time range
    try:
        print("\nAnalyzing recording files in folder...")
        rhd_files = load_rhd_files(args.input, args.recursive)
        if not rhd_files:
            print("No recording files found.")
            exit()
            
        # Process first file to determine date format
        file_example = os.path.basename(rhd_files[0])
        # Store the datetime format to reuse later during processing
        file_datetime_formats = [prompt_user_for_datetime_format(file_example, "file name")]
        
        # Get datetimes from all files
        file_datetimes = []
        for file_path in tqdm(rhd_files, desc="Reading file dates"):
            try:
                # Extract datetime from the file name without reading the whole file
                for fmt in file_datetime_formats:
                    try:
                        file_datetime_str = extract_datetime(os.path.basename(file_path), fmt)
                        file_datetime = datetime.strptime(file_datetime_str, fmt)
                        file_datetimes.append(file_datetime)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logging.debug(f"Error extracting datetime from {file_path}: {e}")
        
        if file_datetimes:
            earliest = min(file_datetimes)
            latest = max(file_datetimes)
            print(f"\nRecordings in folder span from {earliest} to {latest}")
            print(f"Total recording files: {len(file_datetimes)}")
        else:
            print("Could not determine date range from file names.")
    except Exception as e:
        logging.error(f"Error analyzing file dates: {e}")
    
    use_datetime_filter = input("\nDo you want to filter recordings by datetime? (yes/no): ").strip().lower() in ("yes", "y")
    if use_datetime_filter:
        # Prompt for min datetime
        min_datetime_str = args.min_datetime or input("Enter minimum datetime (leave blank for none): ").strip()
        if min_datetime_str:
            try:
                min_datetime = parse_datetime(min_datetime_str)
                print(f"Using minimum datetime: {min_datetime}")
            except ValueError as e:
                print(f"Error: {str(e)}")
                custom_format = input("Enter your datetime format (e.g., %Y-%m-%d %H:%M:%S): ").strip()
                if custom_format:
                    min_datetime = parse_datetime(min_datetime_str, custom_format)
                    print(f"Using minimum datetime: {min_datetime}")
        
        # Prompt for max datetime
        max_datetime_str = args.max_datetime or input("Enter maximum datetime (leave blank for none): ").strip()
        if max_datetime_str:
            try:
                max_datetime = parse_datetime(max_datetime_str)
                print(f"Using maximum datetime: {max_datetime}")
            except ValueError as e:
                print(f"Error: {str(e)}")
                custom_format = input("Enter your datetime format (e.g., %Y-%m-%d %H:%M:%S): ").strip()
                if custom_format:
                    max_datetime = parse_datetime(max_datetime_str, custom_format)
                    print(f"Using maximum datetime: {max_datetime}")
    
    # Interactive channel order input
    channel_order = None
    use_channel_order = input("Do you want to reorder channels? (yes/no): ").strip().lower() in ("yes", "y")
    if use_channel_order:
        # First, let's determine the number of channels from the first recording
        try:
            first_file = load_rhd_files(args.input, args.recursive)[0]
            first_recording = se.read_intan(first_file, stream_id="0", ignore_integrity_checks=True)
            n_channels = first_recording.get_num_channels()
            channel_ids = first_recording.get_channel_ids()
            
            print(f"\nThe recording has {n_channels} channels (0-{n_channels-1}).")
            
            # Show the mapping between indices and channel IDs
            print("\nChannel index to Channel ID mapping:")
            for i, channel_id in enumerate(channel_ids):
                print(f"  Index {i:2d} -> Channel ID: {channel_id}")
            
            print("\nEnter channel order as a Python list of indices (e.g., [1, 0, 3, 2] to swap channels 0/1 and 2/3)")
            print("You don't need to include all channels - missing channels will be dropped from the output.")
            
            while True:
                try:
                    channel_order_input = input("\nChannel order: ").strip()
                    # Evaluate the input as a Python expression (list or array)
                    channel_order = eval(channel_order_input)
                    
                    # Convert to list if it's not already
                    if isinstance(channel_order, (list, tuple, np.ndarray)):
                        channel_order = list(channel_order)
                    else:
                        raise ValueError("Input must be a list, tuple, or array of channel indices")
                    
                    # Basic validation
                    if not all(isinstance(idx, (int, np.integer)) for idx in channel_order):
                        raise ValueError("All elements must be integers")
                    
                    if max(channel_order) >= n_channels or min(channel_order) < 0:
                        raise ValueError(f"Channel indices must be in range [0, {n_channels-1}]")
                    
                    # Show the mapping for the selected order
                    print("\nSelected channel order:")
                    for new_idx, orig_idx in enumerate(channel_order):
                        print(f"  New index {new_idx:2d} <- Original index {orig_idx:2d} (ID: {channel_ids[orig_idx]})")
                    
                    # Confirm with user
                    if input("\nIs this channel order correct? (yes/no): ").strip().lower() in ("yes", "y"):
                        break
                except Exception as e:
                    print(f"Error: {str(e)}. Please try again.")
        except Exception as e:
            logging.error(f"Error determining channel count or parsing channel order: {e}")
            logging.info("Continuing without channel reordering.")
    
    logging.info(f"Using {args.jobs} core(s) for job.")
    # Pass the stored datetime formats to avoid prompting again
    main(args.input, args.output, args.jobs, args.recursive, channel_order, min_datetime, max_datetime, file_datetime_formats)
    multiprocessing.active_children()
