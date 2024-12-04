import logging
import os
import gc
import re
from queue import Queue
from multiprocessing import Process, cpu_count
from datetime import datetime
from tqdm import tqdm
import spikeinterface as si
import spikeinterface.extractors as se
from glob import glob


def progress_updater(total_batches, progress_queue):
    """Progress updater to handle progress bar updates."""
    logging.info("Progress updater started.")
    completed = 0
    with tqdm(total=total_batches, desc="Processing Files") as pbar:
        while completed < total_batches:
            try:
                update = progress_queue.get(timeout=60)  # Wait for signals
                logging.info(f"Progress received: {update}")
                completed += 1
                pbar.update(1)
            except Exception as e:
                logging.error(f"Progress updater failed: {e}")
                break
    logging.info("Progress updater finished.")


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


def producer_task(file_queue, batches, total_batches, progress_queue):
    """Producer task to enqueue batches of files."""
    for i, batch in enumerate(batches):
        file_queue.put(batch)
        progress_queue.put(1)  # Update progress
    file_queue.put(None)  # Signal end of tasks
    logging.info(f"All batches added. Final file_queue size: {file_queue.qsize()}")


def worker_task(file_queue, result_queue, progress_queue):
    """Worker task to process batches of files."""
    logging.info("Worker started.")
    while True:
        logging.info(f"file_queue size before dequeuing: {file_queue.qsize()}")
        batch = file_queue.get()
        if batch is None:
            logging.info("Worker received termination signal.")
            file_queue.put(None)  # Propagate end signal to other workers
            break
        try:
            # Process the batch
            logging.info(f"Processing batch: {batch}")
            recordings = [
                se.read_intan(file, stream_id="0", ignore_integrity_checks=True)
                for file in batch
            ]
            batch_concatenated = si.concatenate_recordings(recordings)
            result_queue.put(batch_concatenated)
            progress_queue.put(1)  # Signal progress
            logging.info("Batch processed successfully.")
        except Exception as e:
            logging.error(f"Error processing batch {batch}: {e}")
        finally:
            del recordings
            gc.collect()
    logging.info("Worker finished.")


def main(input_folder=None, output_folder=None, batch_size=20, num_workers=4):
    """Main function to manage the processing pipeline."""
    # If input folder is not provided, prompt user
    if not input_folder:
        input_folder = input("Enter the input folder containing .rhd files: ").strip()
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")

    # If output folder is not provided, prompt user
    if not output_folder:
        output_folder = input("Enter the output folder for concatenated recording: ").strip()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Recursively find all .rhd files
    file_list = glob(os.path.join(input_folder, "**", "*.rhd"), recursive=True)
    if not file_list:
        raise ValueError("No .rhd files found in the specified input folder.")

    # Prompt user for datetime format using the first file as an example
    datetime_format = prompt_user_for_datetime_format(os.path.basename(file_list[0]))

    # Parse and sort files by datetime
    try:
        file_list = sorted(file_list, key=lambda f: extract_datetime(os.path.basename(f), datetime_format))
    except Exception as e:
        raise ValueError(f"Failed to sort files by datetime: {e}")

    # Divide the file list into batches
    file_batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    num_batches = len(file_batches)
    if num_batches == 0:
        raise ValueError("No files to process after batching.")

    # Create queues
    file_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()

    # Add batches to the file queue
    for batch in file_batches:
        file_queue.put(batch)
    file_queue.put(None)  # Sentinel to signal end of processing

    # Start the progress updater process
    progress_process = Process(target=progress_updater, args=(num_batches, progress_queue))
    progress_process.start()

    # Start worker processes
    workers = [
        Process(target=worker_task, args=(file_queue, result_queue, progress_queue))
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()

    # Collect results
    concatenated_recordings = []
    for _ in range(num_batches):
        result = result_queue.get()
        if result is not None:
            concatenated_recordings.append(result)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Wait for progress updater to finish
    progress_process.join()

    # Final concatenation
    if len(concatenated_recordings) > 1:
        final_recording = si.concatenate_recordings(concatenated_recordings)
    elif concatenated_recordings:
        final_recording = concatenated_recordings[0]
    else:
        raise ValueError("No valid recordings to concatenate.")

    # Save the final concatenated recording
    try:
        final_recording.save(format="binary", folder=output_folder, dtype="int16")
        logging.info(f"Final concatenated recording saved to {output_folder}")
    except Exception as e:
        raise IOError(f"Failed to save the final concatenated recording: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and concatenate .rhd recordings.")
    parser.add_argument("-i", "--input", help="Input folder containing .rhd files.")
    parser.add_argument("-o", "--output", help="Output folder for the concatenated recording.")
    parser.add_argument("-b", "--batch-size", type=int, default=20, help="Number of files per batch.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker processes.")

    args = parser.parse_args()

    main(input_folder=args.input, output_folder=args.output, batch_size=args.batch_size, num_workers=args.workers)
