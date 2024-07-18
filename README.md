# Modular Debiasing of Latent User Representations in Prototype-based Recommender Systems

This repository contains the code and the settings for the submission "Modular Debiasing of Latent User Representations in Prototype-based Recommender Systems" currently under review at ECML-PKDD 2024.
![MMD_Debiasing](./results/plots/gender_lfm2bdemobias_mmd_bacc_ndcg_lams.png "MMD Gender Debiasing")
## Installation

### Environment

- Install the environment with
  `conda env create -f modprotodebias.yml`
- Activate the environment with `conda activate modprotodebias`


### Data


- move into the folder with `cd data/<dataset_folder>`
- run `python <dataset_name>_processor.py`

## Usage

Adjust the configuration of your experiment in `run_full_debiasing.py`.

The experiments can be started with

`python start.py run_full_debiasing`

or define sweep configurations to use with the wandb sweep command

`wandb sweep sweep_config.yaml`