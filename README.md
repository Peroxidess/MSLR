# Codebase for "MSLR: A Self-supervised Representation Learning Method of Tabular Data Based on Multi-scale Ladder Reconstruction"


This directory contains implementations of MSLR framework for self-supervised learning to tabular domain using UCI and MIMIC-IV dataset.


Simply run python3 -m main.py

Note that any model architecture can be used as the encoder and predictor models. 

### Code explanation

(1) preprocess/load_data.py
- Load data

(2) preprocess/get_dataset.py
- Data preprocessing

(3) preprocess/missing_values_imputation.py
- Imputate missing values in dataset

(4) preprocess/representation_learning.py
- Define MSLR framework

(5) model/ae.py
- Define and return the encoder part of MSLR framework

(6) model/baseline.py
- Define supervised classification models

(7) model/evaluate.py
- Performance of computation in reconstruction and prediction tasks

(8) main.py
- Report the prediction performances of entire self-supervised frameworks

(9) arguments.py
- Parameter settings

Note that hyper-parameters should be optimized for different datasets.

## Main Dependency Library
pandas==1.1.5

scikit-learn==0.24.2

torch==1.8.2

torchvision==0.9.2
