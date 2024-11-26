import logging
from pathlib import Path
import pandas as pd
from kilosort.io import load_ops
from processCluster import process_cluster
from sampleToDateTime import load_metadata
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

results_dir = Path("C:/Users/Rudy/Documents/kilosort_out/kilosort4")

logging.info("Loading cluster information and metadata.")

ops = load_ops(results_dir / 'ops.npy')
metadata = load_metadata(results_dir,ops)

cluster_info_path = results_dir / "cluster_info.tsv"
cluster_info = pd.read_csv(cluster_info_path, sep='\t')
cluster_ids = cluster_info['cluster_id'][cluster_info['group'] != 'noise']
logging.info(f"Total clusters to process: {len(cluster_ids)}")

for i, clust_id in enumerate(cluster_ids, start=1):
    logging.info(f"Processing cluster {clust_id} ({i}/{len(cluster_ids)})...")
    process_cluster(clust_id, cluster_info, metadata)

logging.info("Cluster processing complete.")
