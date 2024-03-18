# SpeedE

This repository contains the official source code for the SpeedE model, presented at NAACL 2024 in our paper "SpeedE: Euclidean Geometric Knowledge Graph Embedding Strikes Back".
The repository includes the following:

1. the implementation of SpeedE.
2. the code for training and testing SpeedE on WN18RR, FB15k-237, and YAGO3-10 to reproduce the results presented in our paper (`run_experiments.py`).
3. an `environment.yml` to automatically set up a conda environment with all dependencies.

# Requirements

* Python >= 3.9
* PyKEEN 1.7.0
* Pytorch 1.10.2
* Tensorboard 2.8.0

# Installation

We have provided an `environment.yml` file that can be used to create a conda environment with all required
dependencies. Run `conda env create -f environment.yml` to create the conda environment `Env_SpeedE`.
Afterward, use `conda activate Env_SpeedE` to activate the environment before rerunning our experiments.

# Running SpeedE

Training and evaluation of SpeedE are done by running the `run_experiments.py` script. In particular, a configuration file
must be specified for a SpeedE model, containing all model, training, and evaluation parameters. The best
configuration files for WN18RR and FB15k-237 are provided in the `Best_Configurations` directory and can be adapted to
try out different parameter configurations. To run an experiment, the following parameters need to be specified:

- `config` contains the path to the model configuration (e.g., `config=Best_Configurations/SpeedE/d32_WN18RR.json`)
- `train` contains `true` if the model shall be trained and `false` otherwise.
- `test` contains `true` if the model shall be evaluated on the test set and `false` otherwise.
- `expName` contains the name of the experiment (e.g. `expName=SpeedE_d32_WN18RR`)
- `gpu` contains the id of the gpu that shall be used (e.g., `gpu=0`)
- `seeds` contains the seeds for repeated runs of the experiment (e.g., `seeds=1,2,3`)

Finally, one can run an experiment
with `python run_experiments.py config=<config> train=<true|false> test=<true|false> expName=<expName> gpu=<gpuID> seeds=<seeds>`, 
where angle brackets represent a parameter value. When `test=true`, the evaluation result will be placed under `Benchmarking/final_result`. 

# Reproducing the Results

In the following, we provide the commands to reproduce the results of our paper.

## Low-Dimensional KGC Results

First, we list commands to reproduce our low-dimensional KGC benchmark results (i.e., Table 3).
To execute the upcoming commands, the following variables need to be substituted:

* `<model>` represents the model that shall be trained:
  * `<model>` needs to be substituted by one of [`SpeedE`, `Min_SpeedE`, `ExpressivE`]
* `<dataset>` represents the KGC benchmark dataset: 
  * `<dataset>` needs to be substituted by one of [`WN18RR`, `FB15k-237`, `YAGO3-10`]
  
Train: `python run_experiments.py gpu=0 train=true test=false seeds=1,2,3 config=Best_Configurations/<model>/d32_<dataset>.json expName=<model>_d32_<dataset>`

Test: `python run_experiments.py gpu=0 train=false test=true seeds=1,2,3 config=Best_Configurations/<model>/d32_<dataset>.json expName=<model>_d32_<dataset>`

## Dimensionality Studies

In the following, we list commands to reproduce the results of our dimensionality study on WN18RR (Figures 2 and 3).
To execute the upcoming commands, the following variables need to be substituted:

* `<model>` represents the model that shall be trained:
  * `<model>` needs to be substituted by one of [`SpeedE`, `Min_SpeedE`, `Diff_SpeedE`, `Eq_SpeedE`, `ExpressivE`]
* `<d>` represents the embedding dimensionality: 
  * `<d>` needs to be substituted by one of [`d10`, `d16`, `d20`, `d32`, `d50`, `d200`, `d500`]

Train: `python run_experiments.py gpu=0 train=true test=false seeds=1,2,3 config=Best_Configurations/<model>/<d>_WN18RR.json expName=<model>_<d>_WN18RR`

Test: `python run_experiments.py gpu=0 train=false test=true seeds=1,2,3 config=Best_Configurations/<model>/<d>_WN18RR.json expName=<model>_<d>_WN18RR`

## Training Time per Epoch

In the following, we list commands for measuring the time per epoch of SpeedE, Min_SpeedE, and ExpressivE under `b = n = 500` and `d = 32` (i.e., Table 5).
To execute the upcoming commands, the following variables need to be substituted:

* `<model>` represents the model that shall be trained:
  * `<model>` needs to be substituted by one of [`SpeedE`, `Min_SpeedE`, `ExpressivE`]
* `<dataset>` represents the KGC benchmark dataset: 
  * `<dataset>` needs to be substituted by one of [`WN18RR`, `FB15k-237`, `YAGO3-10`]

We have executed the following commands 
for 500 epochs and reported the average training time per epoch:

`python run_experiments.py gpu=0 train=true test=false seeds=1,2,3 config=Best_Configurations/<model>/b500_n500_d32_<dataset>.json expName=<model>_b500_n500_d32_<dataset>`

# Tensorboard & Convergence Time

The evolution of the model loss and validation metrics (and, thus, also the convergence time, see Table 1) can be observed on tensorboard.
To run tensorboard, execute the provided tensorboard.sh file: `. tensorboard.sh`.

# Citation 

If you use this code or its corresponding paper, please cite our work as follows:

```
@inproceedings{
pavlovic2024speede,
title={SpeedE: Euclidean Geometric Knowledge Graph Embedding Strikes Back},
author={Aleksandar Pavlovi{\'c} and Emanuel Sallinger},
booktitle={Findings of the North American Chapter of the Association for Computational Linguistics},
year={2024}
}
```

# Contact

Aleksandar Pavlović

Research Unit of Databases and Artificial Intelligence

Vienna University of Technology (TU Wien)

Vienna, Austria

<aleksandar.pavlovic@tuwien.ac.at>

# Licenses

The benchmark datasets WN18RR, FB15k-237, and YAGO3-10 are already included in the PyKEEN library. PyKEEN uses the MIT license.
FB15k-237 is a subset of FB15k, which uses the CC BY 2.5 license, and YAGO3-10 is a subset of YAGO3, which uses the CC BY 3.0 license. 
The license of YAGO3-10, FB15k-237, and WN18RR is unknown. This project runs under the MIT license.

Copyright (c) 2024 Aleksandar Pavlović