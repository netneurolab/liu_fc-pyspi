# Benchmarking methods for mapping functional connectivity in the brain
This repository contains code and data in support of "Benchmarking methods for mapping functional connectivity in the brain", now up on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.07.593018v1).

Due to the size limit, the raw data files are not included in this repository. You can download them from the OSF repository [here](https://osf.io/75je2).

## `code`
The [code](code/) folder contains relevant scripts used to run the analyses and generate the figures.

A description of each file follows (in an order that complements the manuscript):

*Scripts in italic* are not necessary for running the analysis scripts as some key variables are provided in [data/derivatives](data/derivatives) folder.

- *[00_run_pyspi_hcp.py](code/00_run_pyspi_hcp.py)* and *[00_submit_job_hcp.sh](code/00_submit_job_hcp.sh)* are sample scripts that run the `pyspi` pipeline on HCP data. *[config-1-minimized.yaml](code/config-1-minimized.yaml)* is the configuration file for the `pyspi` pipeline.
- *[00_sample_generate_data.py](code/00_sample_generate_data.py)* is a sample script that generates clean set of data (used in the main analyses) from raw pyspi outputs.
- [01_similarity.py](code/01_similarity.py) replicates Figure 1.
- [02_network_properties.py](code/02_network_properties.py) replicates Figure 2.
- [03_many_networks.py](code/03_many_networks.py) replicates Figure 3.
- [04_individual_differences.py](code/04_individual_differences.py) replicates Figure 4.
  - *[04_sample_generate_fingerprinting.py](code/04_sample_generate_fingerprinting.py)* is a sample script that generates fingerprinting data from raw pyspi outputs.
  - *[04_sample_generate_prediction.py](code/04_sample_generate_prediction.py)* is a sample script that generates brain-behavior prediction data from raw pyspi outputs.
- [05_information_theory.py](code/05_information_theory.py) replicates Figure 5.
  - *[05_sample_generate_phiid.m](code/05_sample_generate_phiid.m)* is a sample script that generates integrated information decomposition results from HCP data. The original script was taken from [here](https://www.sciencedirect.com/science/article/pii/S1053811923000745), however, we recommend using the updated versions recommended in [(Luppi et al., 2024)](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(23)00284-X).
  - *[05_sample_generate_stats.py](code/05_sample_generate_stats.py)* is a sample script that generates dominance analysis results.
- [06_sensitivity.py](code/06_sensitivity.py) replicates Figure 6.
- [utils.py](code/utils.py) contains utility functions used in the analyses.


## `data`

The [data](data/) folder contains data files used for the analyses. Please download files you need from the OSF repository [here](https://osf.io/75je2) for replication.

- [data/derivatives](data/derivatives) contains some key variables used in the analyses. You probably only need these to run the analysis scripts in the [code](code/) folder.
- [data/raw](data/raw) contains raw pyspi outputs and some additional data files.
