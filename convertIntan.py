import logging
import os
import gc
import re
from queue import Queue
from multiprocessing import Process, Queue as MPQueue, cpu_count
from datetime import datetime
from tqdm import tqdm
import spikeinterface as si
import spikeinterface.extractors as se


def progress_updater(total_batches, progress_queue):
    """Progress updater to handle progress bar updates."""
    with tqdm(total=total_batches, desc="Processing Files") as pbar:
        for _ in range(total_batches):
            progress_queue.get()  # Wait for a signal from workers
            pbar.update(1)


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
        progress_queue.put((i + 1, total_batches))  # Update progress
    file_queue.put(None)  # Signal end of tasks

def worker_task(file_queue, result_queue, progress_queue):
    """Worker task to process batches of files."""
    while True:
        batch = file_queue.get()
        if batch is None:
            file_queue.put(None)  # Propagate end signal to other workers
            break
        recordings = None  # Initialize to avoid reference errors
        try:
            # Read files and concatenate the recordings
            recordings = [
                se.read_intan(file, stream_id="0", ignore_integrity_checks=True)
                for file in batch
            ]
            batch_concatenated = si.concatenate_recordings(recordings)
            result_queue.put(batch_concatenated)  # Send result to result queue
        except Exception as e:
            logging.error(f"Error processing batch {batch}: {e}")
        finally:
            # Free memory and signal progress
            if recordings is not None:
                del recordings
            gc.collect()
            progress_queue.put(1)  # Signal batch completion to progress updater


def final_stage_task(result_queue, final_result_path, num_batches, progress_queue):
    """Final stage to concatenate intermediate results."""
    intermediate_results = []
    for _ in range(num_batches):
        result = result_queue.get()
        intermediate_results.append(result)
        progress_queue.put(1)  # Update progress for final stage
    try:
        final_recording = si.concatenate_recordings(intermediate_results)
        final_recording.save(folder=final_result_path, dtype="int16", format="binary")
        print("Final concatenation complete!")
    except Exception as e:
        print(f"Error during final concatenation: {e}")
    finally:
        del intermediate_results  # Free memory
        gc.collect()

def progress_bar_task(progress_queue, total_tasks, desc):
    """Task to update a progress bar."""
    with tqdm(total=total_tasks, desc=desc, position=0) as pbar:
        completed = 0
        while completed < total_tasks:
            update = progress_queue.get()
            if isinstance(update, tuple):
                # Producer updates with current batch number and total batches
                completed = update[0]
                pbar.n = completed
                pbar.total = update[1]
            else:
                # Worker updates with completed task increments
                completed += update
            pbar.update(0)

def main(input_folder, batch_size, num_workers, output_file):
    """Main function to manage the processing pipeline."""
    # Load all .rhd files
    file_list = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".rhd")
    ]
    file_list.sort()  # Sort files to preserve chronological order if filenames have datetime

    # Divide the file list into batches
    file_batches = [file_list[i:i + batch_size] for i in range(0, len(file_list), batch_size)]
    num_batches = len(file_batches)

    if num_batches == 0:
        logging.error("No files found for processing.")
        return

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
        logging.error("No valid recordings to concatenate.")
        return

    # Save the final concatenated recording
    final_recording.save(format="binary", folder=output_file)
    logging.info(f"Final concatenated recording saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and concatenate .rhd recordings.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing .rhd files.")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the concatenated recording.")
    parser.add_argument("-b", "--batch-size", type=int, default=20, help="Number of files per batch.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker processes.")

    args = parser.parse_args()

    main(args.input, args.batch_size, args.workers, args.output)