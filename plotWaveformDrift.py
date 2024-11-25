import logging
from pathlib import Path
import pandas as pd
from kilosort.io import load_ops
from dateutil import parser
from processCluster import process_cluster
from sampleToDateTime import load_metadata
from types import SimpleNamespace
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

results_dir = Path("C:/Users/Rudy/Documents/kilosort_out/kilosort4")

# Load data
logging.info("Loading cluster information and metadata.")
cluster_info_path = results_dir / "cluster_info.tsv"
metadata_path = results_dir / "../recording_metadata.json"

cluster_info = pd.read_csv(cluster_info_path, sep='\t')
cluster_ids = cluster_info['cluster_id'][cluster_info['group'] != 'noise']
ops = load_ops(results_dir / 'ops.npy')

metadata = SimpleNamespace()
metadata.data = load_metadata(metadata_path)
metadata.tstart = parser.parse(metadata.data[0]["datetime"])
metadata.chanMap = ops['chanMap']
metadata.results_dir = results_dir
metadata.th_time = 3600
metadata.cmap = 'RdYlGn'
metadata.th_spikes = 10

logging.info(f"Total clusters to process: {len(cluster_ids)}")

# Process clusters
for i, clust_id in enumerate(cluster_ids, start=1):
    logging.info(f"Processing cluster {clust_id} ({i}/{len(cluster_ids)})...")
    try:
        process_cluster(clust_id, cluster_info, metadata)
        logging.info(f"Successfully processed cluster {clust_id}.")
    except Exception as e:
        logging.error(f"Error processing cluster {clust_id}: {e}")

logging.info("Cluster processing complete.")
