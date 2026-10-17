# Predicting Building Damage in the Nepal (Gorkha) Earthquake

UE24CS352A - Machine Learning mini-project. Predict the damage grade (1 = low, 2 = medium,
3 = near-complete destruction) of buildings after the April 2015 Nepal earthquake, using
**K-Nearest Neighbours**, a **from-scratch NumPy Neural Network** and a **Random Forest**.
Based on the DrivenData competition *Richter's Predictor: Modeling Earthquake Damage*.

**Team:** [HARSHITH KK- PES1UG25AM817], [Member 2 - SRN] | Section [X]

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the three CSVs from the [DrivenData competition page](https://www.drivendata.org/competitions/57/nepal-earthquake/data/)
(`train_values.csv`, `train_labels.csv`, `test_values.csv`) and place them in `data/`.

## Run

```bash
python run_all.py --data_dir data --team "Name1 (SRN1), Name2 (SRN2) | Sec A"   # full pipeline
python run_all.py --fast                                                       # ~15% subsample, quick demo
python -m src.train --data_dir data                                            # training only
```

Outputs: `results/metrics.json`, `results/*.png` (comparison, confusion matrices, feature importance,
NN loss), `results/submission.csv` (upload to DrivenData), plus `writeup.pdf` and `slides.pptx`,
regenerated automatically from the measured metrics.

Smoke test without the real data: `python tests/make_synthetic.py --out data_synth && python -m src.train --data_dir data_synth`
(synthetic numbers are meaningless - never report them).

## Method

| Model | Pipeline |
|---|---|
| KNN | one-hot -> StandardScaler -> SelectKBest(f_classif, 20) -> KNN (k = 7) |
| Neural net | one-hot -> StandardScaler -> 2-layer NN (sigmoid / softmax, L2, mini-batch 1000, hidden = round(sqrt(n_in*n_out)), lr = 2, 3 restarts) |
| Random Forest | ordinal encoding -> RandomForest, GridSearchCV(3-fold) over `min_samples_leaf` {1,5}, `n_estimators` {50,100} |

Evaluation: stratified 80/20 split of the labelled data (the DrivenData test labels are hidden);
metric = micro-averaged F1 (also macro F1, confusion matrices, train-vs-holdout gap, fit time).
The best model is refit on all labelled data to create the submission.

## Structure

```
src/data.py  src/neural_net.py  src/models.py  src/train.py
docs/build_writeup.py  docs/build_slides.py  docs/common.py
tests/make_synthetic.py   run_all.py   requirements.txt
```

## Reference
Y. Liang, *Application of Machine Learning Methods to Predict the Level of Buildings Damage in the
Nepal Earthquake*, Stanford CS229 (2021) - the approach we reproduce and extend with a proper holdout.
