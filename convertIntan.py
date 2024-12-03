import os
import gc
import re
from queue import Queue
from multiprocessing import Process, Queue as MPQueue, cpu_count
from datetime import datetime
from tqdm import tqdm
import spikeinterface as si
import spikeinterface.extractors as se

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
    """Worker task to process batches."""
    while True:
        batch = file_queue.get()
        if batch is None:
            file_queue.put(None)  # Propagate end signal to other workers
            break
        try:
            # Process the batch
            recordings = [
                se.read_intan(file, stream_id="0", ignore_integrity_checks=True)
                for file in batch
            ]
            batch_concatenated = si.concatenate_recordings(recordings)
            result_queue.put(batch_concatenated)
        except Exception as e:
            print(f"Error processing batch {batch}: {e}")
        finally:
            del recordings  # Free memory
            gc.collect()
            progress_queue.put(1)  # Update progress for batch completion

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

def main(input_folder, output_folder, batch_size, num_workers):
    # Find all RHD files
    rhd_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files
        if file.endswith(".rhd")
    ]

    if not rhd_files:
        print("No .rhd files found in the specified folder.")
        return

    # Prompt user for datetime format and sort files by datetime
    datetime_format = prompt_user_for_datetime_format(os.path.basename(rhd_files[0]))
    try:
        rhd_files = sorted(
            rhd_files,
            key=lambda x: extract_datetime(os.path.basename(x), datetime_format),
        )
    except ValueError as e:
        print(f"Error parsing datetime: {e}")
        return

    # Split files into batches
    batches = [rhd_files[i:i + batch_size] for i in range(0, len(rhd_files), batch_size)]
    total_batches = len(batches)

    # Queues for inter-process communication
    file_queue = MPQueue()
    result_queue = MPQueue()
    progress_queue = MPQueue()

    # Start progress bar process
    total_tasks = total_batches + (total_batches - 1)  # Batches + final concatenation steps
    progress_bar = Process(target=progress_bar_task, args=(progress_queue, total_tasks, "Processing Files"))
    progress_bar.start()

    # Start producer process
    producer = Process(target=producer_task, args=(file_queue, batches, total_batches, progress_queue))
    producer.start()

    # Start worker processes
    workers = [
        Process(target=worker_task, args=(file_queue, result_queue, progress_queue))
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()

    # Start final stage process
    final_stage = Process(target=final_stage_task, args=(result_queue, output_folder, total_batches, progress_queue))
    final_stage.start()

    # Wait for all processes to finish
    producer.join()
    for worker in workers:
        worker.join()
    final_stage.join()
    progress_bar.join()

    print("All tasks completed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flexible and scalable RHD file concatenation pipeline with datetime sorting and progress bar.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing RHD files.")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the concatenated recording.")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size for processing files.")
    parser.add_argument("-w", "--workers", type=int, default=cpu_count(), help="Number of worker processes.")
    args = parser.parse_args()

    main(args.input, args.output, args.batch_size, args.workers)
