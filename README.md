# Kilosort Scripts

A collection of Python utilities for getting our data to work with Kilosort. 

## Overview

This repository contains two main tools:

1. Convert Intan (.rhd) recordings to binary format that Kilosort can read
2. Create summary PDF reports for Kilosort sorted clusters


## Kilosort Workflow

The full spike sorting pipeline involves several steps, from raw data to sorted units:

### 1. Convert Intan Recordings to Binary Format

First, convert your .rhd files to a single binary .raw file that Kilosort can process:

```bash
python scripts/convertIntan.py
```

This will guide you through selecting files, mapping channels, and applying preprocessing. The output includes a binary .raw file and configuration files for Kilosort.

### 2. Run Kilosort on Your Data

#### 2.1 Install Kilosort

##### Scenario 1: Small Datasets (CPU install)

For smaller recordings (a few minutes to test), you can run Kilosort locally:

```bash
bash .requirements/req-install.sh # use option 1
```

##### Scenario 2: Large Datasets (Cluster GPU)

For larger recordings, you will likely need to use the cluster:

1. Log into OpenOnDemand: https://ood.rc.fas.harvard.edu
2. Start a GPU session:
   - Click "Remote Desktop"
   - Request 1 GPU, enough memory to hold your data, and 4+ CPUs
   - Set your working directory
   - Click "Launch"

3. When your session starts, open up a terminal
   
4. install some kind of conda/mamba (micromamba is my goto on the cluster)
5. ```bash 
    bash .requirements/req-install.sh # use option 2
   ```

#### 2.2 Run Kilosort and start clustering
1. Launch the GUI: ```python -m kilosort```
2. Load in the .raw file
3. Select save directory
4. Define the probe geometry (see tips below)
5. Run!

### 3. Review Sorting Results in Phy

After Kilosort finishes, review and curate the results in Phy:
#### 3.1 Install phy2 (and environment)
 ```bash 
 # create new conda/mamba environment
conda create -n phy2

# activate new environment
conda activate phy2

# install phy environment
bash .requirements/req-install.sh # use option 3
   ```


1. Open your results in Phy:
   ```bash
   cd kilosort-results-folder
   phy template-gui params.py
   ```

2. In Phy:
   - Review clusters and waveforms
   - Split/merge clusters as needed
   - Save your changes frequently

### 4. Generate Cluster Summary Reports

After curation, create PDF reports of your sorted units:

```bash
python scripts/createClusterPDF.py --results-dir /path/to/kilosort/results
```

To implement a voltage threshold (e.g., 30μV) to filter out noise:

```bash
python scripts/createClusterPDF.py --results-dir /path/to/kilosort/results --min-amplitude 30
```

## Scripts

### convertIntan.py

This script converts your Intan (.rhd) recordings to a binary format that Kilosort can read. It handles the concatenation of multiple recording files and applies preprocessing like common referencing and bandpass filtering. You can reorder channels to match your probe configuration, and filter recordings by date and time. The script also uses multiple cores to speed up the conversion.

#### Running the Script

There are two ways to run this script, depending on your environment:

**Interactive Mode**

If you've got a terminal where you can type responses (like your laptop, an salloc session, etc.), just run:

```bash
python scripts/convertIntan.py
```

The script will walk you through everything - where to find files, how to map channels, datetime handling, etc. 

**Headless/Non-interactive Mode**

For batch jobs, clusters, or anywhere you can't provide interactive input:

```bash
python scripts/convertIntan.py \
  --input /path/to/rhd_files \
  --output /path/to/output_folder \
  --channel-order "[0,1,2,3,4,5,6,7]" \
  --force \
  --noninteractive
```

You'll need to specify all params upfront. Use `--force` to overwrite existing files without asking, and `--datetime-format` if you have non-standard date formats. For date filtering, add `--min-datetime` and `--max-datetime`.

### createClusterPDF.py

This script generates visual reports of your Kilosort results. It creates a PDF with template waveforms, amplitude stability plots, and ISI histograms for each sorted unit. Great for quickly assessing spike sorting quality.

Basic usage:

```bash
python scripts/createClusterPDF.py --results-dir /path/to/kilosort/results
```

If you want to implement a voltage threshold to get rid of noise (eg. 30uV):

```bash
python scripts/createClusterPDF.py --results-dir /path/to/kilosort/results --min-amplitude 30
```