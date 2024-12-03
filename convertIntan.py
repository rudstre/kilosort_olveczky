import logging
import os
import gc
import re
import argparse
from queue import Queue
from multiprocessing import Process, Queue as MPQueue, cpu_count
from datetime import datetime

import psutil
import spikeinterface as si
import spikeinterface.extractors as se


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize previous counters
prev_read_bytes = 0
prev_write_bytes = 0


def get_process_memory_usage():
    """Get memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / (1024**2)  # Convert to MB
    for child in process.children(recursive=True):
        mem_usage += child.memory_info().rss / (1024**2)
    return mem_usage


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

    logging.info(f"Memory Usage: {memory / 1000:.2f}GB")
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


def producer_task(file_queue, batches):
    """Producer task to enqueue batches of files."""
    for batch in batches:
        file_queue.put(batch)
    file_queue.put(None)  # Signal end of tasks


def worker_task(file_queue, result_queue):
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


def final_stage_task(result_queue, final_result_path, num_batches):
    """Final stage to concatenate intermediate results."""
    intermediate_results = []
    for _ in range(num_batches):
        result = result_queue.get()
        intermediate_results.append(result)
    try:
        final_recording = si.concatenate_recordings(intermediate_results)
        final_recording.save(folder=final_result_path, dtype="int16", format="binary")
        print("Final concatenation complete!")
    except Exception as e:
        print(f"Error during final concatenation: {e}")
    finally:
        del intermediate_results  # Free memory
        gc.collect()


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

    # Queues for inter-process communication
    file_queue = MPQueue()
    result_queue = MPQueue()

    # Start producer process
    producer = Process(target=producer_task, args=(file_queue, batches))
    producer.start()

    # Start worker processes
    workers = [
        Process(target=worker_task, args=(file_queue, result_queue))
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()

    # Start final stage process
    final_stage = Process(target=final_stage_task, args=(result_queue, output_folder, len(batches)))
    final_stage.start()

    # Wait for all processes to finish
    producer.join()
    for worker in workers:
        worker.join()
    final_stage.join()

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible and scalable RHD file concatenation pipeline with datetime sorting.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing RHD files.")
    parser.add_argument("-o", "--output", required=True, help="Output folder for the concatenated recording.")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size for processing files.")
    parser.add_argument("-w", "--workers", type=int, default=cpu_count(), help="Number of worker processes.")
    args = parser.parse_args()

    main(args.input, args.output, args.batch_size, args.workers)
