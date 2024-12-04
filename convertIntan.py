import os
import logging
import multiprocessing
from multiprocessing import Queue
from tqdm import tqdm
import spikeinterface as si
import spikeinterface.extractors as se
import gc
import re
from datetime import datetime


def build_regex_from_format(datetime_format):
    """Convert a datetime format string into a regex pattern."""
    format_to_regex = {
        "%Y": r"\d{4}", "%y": r"\d{2}", "%m": r"\d{2}", "%d": r"\d{2}",
        "%H": r"\d{2}", "%M": r"\d{2}", "%S": r"\d{2}",
    }
    regex_pattern = ""
    i = 0
    while i < len(datetime_format):
        if datetime_format[i:i + 2] in format_to_regex:
            regex_pattern += format_to_regex[datetime_format[i:i + 2]]
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
    return datetime.strptime(match.group(), datetime_format)


def prompt_user_for_datetime_format(example_name):
    """Prompt the user to enter the datetime format based on an example name."""
    print(
        f"Provide the datetime format for file names.\n"
        f"Example file name: {example_name}\n"
        "Use Python datetime codes (e.g., '%y%m%d_%H%M%S' for '240830_145124')."
    )
    while True:
        datetime_format = input("Enter the datetime format: ").strip()
        try:
            if re.search(build_regex_from_format(datetime_format), example_name):
                return datetime_format
        except ValueError:
            pass
        print(f"Invalid format '{datetime_format}'. Try again.")


def worker_task(batch, progress_queue):
    """Worker task to process a batch of files."""
    try:
        recordings = [
            se.read_intan(file, stream_id="0", ignore_integrity_checks=True)
            for file in batch
        ]
        batch_concatenated = si.concatenate_recordings(recordings)
        progress_queue.put(1)  # Update progress
        return batch_concatenated
    except Exception as e:
        logging.error(f"Error processing batch {batch}: {e}")
        return None
    finally:
        # Explicit garbage collection
        del recordings
        gc.collect()


def process_batches_in_parallel(file_batches, num_workers):
    """Process batches using multiprocessing Pool."""
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        results = []
        total_batches = len(file_batches)

        with tqdm(total=total_batches, desc="Processing Files") as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                # Launch worker tasks
                tasks = [
                    pool.apply_async(worker_task, args=(batch, progress_queue))
                    for batch in file_batches
                ]

                # Collect results and update progress
                for task in tasks:
                    result = task.get()
                    if result is not None:
                        results.append(result)
                    # Update progress bar
                    while not progress_queue.empty():
                        progress_queue.get()
                        pbar.update(1)

                # Explicit garbage collection after each batch
                gc.collect()

        return results


def main(input_dir, batch_size, num_workers, output_file):
    """Main function to manage the processing pipeline."""
    # Recursively find .rhd files
    file_list = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.endswith(".rhd")
    ]
    if not file_list:
        logging.error("No .rhd files found in the input folder.")
        return

    # Prompt for datetime format using the first file as an example
    datetime_format = prompt_user_for_datetime_format(file_list[0])

    # Extract datetime from filenames and sort the file list
    try:
        file_list = sorted(
            file_list,
            key=lambda x: extract_datetime(os.path.basename(x), datetime_format),
        )
    except Exception as e:
        logging.error(f"Error parsing datetime: {e}")
        return

    # Divide the file list into batches
    file_batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]

    # Process batches in parallel
    logging.info(f"Processing {len(file_batches)} batches with {num_workers} workers...")
    batch_results = process_batches_in_parallel(file_batches, num_workers)

    # Final concatenation
    if len(batch_results) > 1:
        final_recording = si.concatenate_recordings(batch_results)
    elif batch_results:
        final_recording = batch_results[0]
    else:
        logging.error("No valid recordings to concatenate.")
        return

    # Save the final concatenated recording
    final_recording.save(format="binary", folder=output_file, dtype="int16")
    logging.info(f"Final concatenated recording saved to {output_file}")

    # Final garbage collection
    del batch_results, final_recording
    gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and concatenate .rhd recordings.")
    parser.add_argument("-i", "--input", help="Input folder containing .rhd files.")
    parser.add_argument("-o", "--output", help="Output folder for the concatenated recording.")
    parser.add_argument("-b", "--batch-size", type=int, default=20, help="Number of files per batch.")
    parser.add_argument("-w", "--workers", type=int, default=48, help="Number of worker processes.")

    args = parser.parse_args()
    input_folder = args.input or input("Enter input folder: ").strip()
    output_folder = args.output or input("Enter output folder: ").strip()

    main(input_folder, args.batch_size, args.workers, output_folder)
