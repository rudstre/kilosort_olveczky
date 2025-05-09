#!/usr/bin/env python3
"""
Convert Intan (.rhd) recordings to binary format with preprocessing.

This script concatenates and preprocesses Intan .rhd files, with options for:
- Channel reordering
- Datetime filtering
- Common reference and bandpass filtering

Usage:
    python convert_intan_to_binary.py \
      --input /path/to/rhd_files \
      --output /path/to/output_folder \
      [--recursive] \
      [--jobs 4] \
      [--min-datetime "2025-05-01 00:00:00"] \
      [--max-datetime "2025-05-08 23:59:59"] \
      [--datetime-format "%y%m%d_%H%M%S"] \
      [--channel-order "[0,1,2,3]"] \
      [--force] \
      [--noninteractive] \
      [--verbose]
"""
import sys
import re
import json
import shutil
import logging
import argparse
import ast
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse as dateutil_parse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, freeze_support
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from typing import List, Tuple, Optional, Dict, Any

# Configure root logger once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prompt_user_for_datetime_format(example: str, source: str) -> str:
    fmt = input(f"Couldn't detect a datetime in {source} '{example}'. Enter a format (e.g. '%y%m%d_%H%M%S'): ")
    return fmt.strip()


def prompt_for_channel_order(input_folder: Path, recursive: bool) -> List[int]:
    while True:
        raw = input("Enter channel order as a Python list (e.g. [0,1,2,3]): ")
        try:
            order = ast.literal_eval(raw)
            if isinstance(order, (list, tuple)) and all(isinstance(x, int) for x in order):
                return list(order)
        except Exception:
            pass
        print("Invalid format, please try again.")


def yes_no(prompt: str) -> bool:
    resp = input(f"{prompt} (yes/no): ").strip().lower()
    return resp in ('yes', 'y')


def parse_channel_order_arg(s: str) -> List[int]:
    try:
        order = ast.literal_eval(s)
    except Exception:
        raise argparse.ArgumentTypeError("Channel order must be a Python list of integers")
    if not isinstance(order, (list, tuple)) or not all(isinstance(x, int) for x in order):
        raise argparse.ArgumentTypeError("Channel order must be a list of integers")
    return list(order)


def build_regex_from_format(datetime_format: str) -> str:
    mapping = {"%Y": r"\d{4}", "%y": r"\d{2}", "%m": r"\d{2}", "%d": r"\d{2}",
               "%H": r"\d{2}", "%M": r"\d{2}", "%S": r"\d{2}"}
    pattern = ""
    i = 0
    while i < len(datetime_format):
        token = datetime_format[i : i + 2]
        if token in mapping:
            pattern += mapping[token]
            i += 2
        else:
            pattern += re.escape(datetime_format[i])
            i += 1
    return pattern


def extract_datetime_from_name(name: str, formats: List[str]) -> datetime:
    try:
        return dateutil_parse(name, fuzzy=True)
    except ValueError:
        pass
    for fmt in formats:
        regex = build_regex_from_format(fmt)
        match = re.search(regex, name)
        if match:
            try:
                return datetime.strptime(match.group(), fmt)
            except ValueError:
                continue
    raise ValueError(f"No datetime matching formats {formats} found in '{name}'")


def parse_datetime(datetime_str: str, format_str: Optional[str] = None) -> datetime:
    if not datetime_str:
        raise ValueError("Empty datetime string")
    if format_str:
        try:
            return datetime.strptime(datetime_str, format_str)
        except ValueError:
            pass
    for fmt in ("%y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse datetime: {datetime_str}")


def sanitize_path(path_str: str) -> Path:
    cleaned = path_str.strip().strip('"').strip("'")
    return Path(cleaned).expanduser().resolve()


def load_rhd_files(input_folder: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.rhd" if recursive else "*.rhd"
    files = list(input_folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .rhd files found in {input_folder} (recursive={recursive})")
    return files


def validate_output_folder(output_folder: Path, force: bool, interactive: bool) -> None:
    if output_folder.exists():
        if force:
            shutil.rmtree(output_folder)
            logger.info(f"Deleted existing output folder: {output_folder}")
        elif interactive and yes_no(f"Output folder '{output_folder}' exists. Delete it?"):
            shutil.rmtree(output_folder)
            logger.info(f"Deleted existing output folder: {output_folder}")
        else:
            raise RuntimeError("Existing output folder not cleared")


def filter_files_by_datetime(
    rhd_files: List[Path],
    datetime_formats: List[str],
    min_dt: Optional[datetime],
    max_dt: Optional[datetime],
) -> List[Path]:
    if not (min_dt or max_dt):
        return rhd_files
    filtered = []
    skipped = 0
    for f in tqdm(rhd_files, desc="Filtering files by datetime"):
        try:
            dt = extract_datetime_from_name(f.name, datetime_formats)
            if (min_dt and dt < min_dt) or (max_dt and dt > max_dt):
                skipped += 1
                continue
        except ValueError:
            pass
        filtered.append(f)
    logger.info(f"Skipped {skipped} files outside datetime range; {len(filtered)} remain.")
    return filtered


def process_rhd_file(args: Tuple[Path, List[str]]) -> Tuple[Optional[si.BaseRecording], Dict[str, Any]]:
    file_path, datetime_formats = args
    try:
        rec = se.read_intan(str(file_path), stream_id="0", ignore_integrity_checks=True)
        dt = extract_datetime_from_name(file_path.name, datetime_formats)
        frames = rec.get_num_frames()
        meta = {
            "file_name": file_path.name,
            "start_sample": 0,
            "end_sample": frames - 1,
            "datetime": dt.isoformat()
        }
        return rec, meta
    except (OSError, ValueError) as e:
        logger.exception(f"Failed processing {file_path}")
        return None, {"file_name": file_path.name, "error": str(e)}


def reorder_channels(rec: si.BaseRecording, channel_order: List[int]) -> si.BaseRecording:
    all_ids = rec.get_channel_ids()
    n = rec.get_num_channels()
    if any(idx < 0 or idx >= n for idx in channel_order):
        raise ValueError(f"Channel indices must be between 0 and {n-1}")
    ids = [all_ids[i] for i in channel_order]
    return rec.channel_slice(ids)


def preprocess_recording(rec: si.BaseRecording) -> si.BaseRecording:
    logger.info("Applying bandpass filter and common reference")
    band = sp.bandpass_filter(recording=rec, freq_min=300, freq_max=6000)
    return sp.common_reference(recording=band, operator="median")


def create_processing_metadata(
    min_dt: Optional[datetime],
    max_dt: Optional[datetime],
    channel_order: Optional[List[int]],
) -> Dict[str, Any]:
    return {
        "datetime_range": {
            "min_datetime": min_dt.isoformat() if min_dt else None,
            "max_datetime": max_dt.isoformat() if max_dt else None
        },
        "processing_info": {
            "channel_order_used": bool(channel_order),
            "channel_order": channel_order or []
        }
    }


def gather_parameters() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Convert Intan .rhd recordings to binary format.")
    parser.add_argument('-i', '--input',           type=Path,                   help='Input folder containing recordings')
    parser.add_argument('-o', '--output',          type=Path,                   help='Output folder for processed files')
    parser.add_argument('-j', '--jobs',            type=int, default=multiprocessing.cpu_count(), help='Parallel jobs')
    parser.add_argument('-r', '--recursive',       action='store_true',         help='Search subfolders for recordings')
    parser.add_argument('--min-datetime',          type=str,                   help='Minimum datetime to include')
    parser.add_argument('--max-datetime',          type=str,                   help='Maximum datetime to include')
    parser.add_argument('--datetime-format',       action='append', type=str,  help='Filename datetime format; repeatable')
    parser.add_argument('--channel-order',         type=parse_channel_order_arg, help='List of channel indices')
    parser.add_argument('--force',                 action='store_true',         help='Delete existing output without prompting')
    parser.add_argument('-y','--noninteractive',   action='store_true',         help='Run without prompts; error on missing args')
    parser.add_argument('--verbose',               action='store_true',         help='Enable debug logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.noninteractive and (not args.input or not args.output):
        parser.error("--noninteractive requires --input and --output")

    interactive = not args.noninteractive

    # Input folder
    if args.input:
        input_path = args.input.resolve()
    elif interactive:
        while True:
            candidate = sanitize_path(input("Enter path to input folder: "))
            if candidate.is_dir():
                input_path = candidate
                break
            print(f"Invalid folder: {candidate}")
    else:
        parser.error("Missing --input")

    # Output folder
    if args.output:
        output_path = args.output.resolve()
    elif interactive:
        output_path = sanitize_path(input("Enter path for output folder: "))
    else:
        parser.error("Missing --output")

    # Recursive?
    recursive = args.recursive
    if interactive and not args.recursive:
        recursive = yes_no("Search for recordings in subfolders?")

    # Datetime formats
    formats = args.datetime_format or []
    if not formats:
        sample_name = load_rhd_files(input_path, recursive)[0].name
        default_fmt = "%y%m%d_%H%M%S"
        if re.search(build_regex_from_format(default_fmt), sample_name):
            logger.info(f"Using default datetime format '{default_fmt}'")
            formats = [default_fmt]
        elif interactive:
            formats = [prompt_user_for_datetime_format(sample_name, "filename")]
        else:
            parser.error("Could not infer --datetime-format; specify it or run interactively")

    def get_optional_datetime(arg_str: Optional[str], question: str) -> Optional[datetime]:
        if arg_str:
            return parse_datetime(arg_str)
        if interactive and yes_no(question):
            return parse_datetime(input("Enter datetime: "))
        return None

    min_dt = get_optional_datetime(args.min_datetime, "Filter by minimum datetime?")
    max_dt = get_optional_datetime(args.max_datetime, "Filter by maximum datetime?")

    if args.channel_order is not None:
        ch_order = args.channel_order
    elif interactive and yes_no("Reorder channels?"):
        ch_order = prompt_for_channel_order(input_path, recursive)
    else:
        ch_order = None

    return {
        "input_folder": input_path,
        "output_folder": output_path,
        "n_jobs": args.jobs,
        "recursive": recursive,
        "datetime_formats": formats,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "channel_order": ch_order,
        "force": args.force,
        "interactive": interactive,
    }


def main(
    input_folder: Path,
    output_folder: Path,
    n_jobs: int,
    recursive: bool,
    datetime_formats: List[str],
    min_dt: Optional[datetime],
    max_dt: Optional[datetime],
    channel_order: Optional[List[int]],
    force: bool,
    interactive: bool,
) -> None:
    files = load_rhd_files(input_folder, recursive)
    files = filter_files_by_datetime(files, datetime_formats, min_dt, max_dt)
    if not files:
        raise RuntimeError("No .rhd files to process after filtering.")

    # only prompt for deletion if interactive
    validate_output_folder(output_folder, force, interactive)

    tasks = [(f, datetime_formats) for f in files]
    results: List[Tuple[Optional[si.BaseRecording], Dict[str, Any]]] = []

    with Pool(processes=n_jobs) as pool:
        for rec, meta in tqdm(pool.imap_unordered(process_rhd_file, tasks),
                              total=len(tasks), desc="Reading files"):
            if rec is not None:
                results.append((rec, meta))

    if not results:
        raise RuntimeError("All file processing failed.")

    # Sort & accumulate
    results.sort(key=lambda rm: rm[1]["datetime"])
    recs, metas = zip(*results)
    cum = 0
    updated = []
    for r, m in zip(recs, metas):
        frames = r.get_num_frames()
        m["start_sample"] = cum
        cum += frames
        m["end_sample"] = cum - 1
        updated.append(m)

    final_rec = si.concatenate_recordings(list(recs)) if len(recs) > 1 else recs[0]
    if channel_order:
        final_rec = reorder_channels(final_rec, channel_order)
    processed = preprocess_recording(final_rec)

    logger.info(f"Saving processed recording to {output_folder}")
    processed.save(dtype="int16", format="binary", folder=str(output_folder), n_jobs=n_jobs)

    all_meta = {
        "files": updated,
        "run_info": create_processing_metadata(min_dt, max_dt, channel_order),
    }
    with open(output_folder / "recording_metadata.json", 'w') as mf:
        json.dump(all_meta, mf, indent=4)
    logger.info(f"Metadata written to {output_folder / 'recording_metadata.json'}")


if __name__ == "__main__":
    # Support Windows multiprocessing safely
    freeze_support()
    try:
        params = gather_parameters()
        main(**params)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)