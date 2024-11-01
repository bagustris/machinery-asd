# Anomaly sound detection with CCC Loss function

## Downloading dataset

First, you need to download the IDMT and MIMII datasets. I located these data
inside `data` directory. If you locate them elsewhere, you need to adjust those paths (in `baseline.py`).

Link for download:  

- IDMT: <https://zenodo.org/record/7551261>
- MIMII Pump: <https://www.kaggle.com/datasets/senaca/mimii-pump-sound-dataset>

## Installation
The code is tested to work on Python 3.8; Python versions higher than 3.8 should work, too, although it is not tested. The following requirements for installation are only to make it work with my GPU and Python versions. As long as all required libraries can be installed, there should be no problem with running the program.

```bash
pip install -r requirements.txt # gpu
pip install -r requirements-cpu.txt # cpu
```

## Running the code

IDMT works out of the box with default MSE loss. You only need to run `baseline4.py`.

```bash
$ python baseline5.py
...
The error threshold is set to be:  100.9849967956543
              precision    recall  f1-score   support

      Normal       0.99      0.70      0.82       669
     Anomaly       0.77      0.99      0.87       665

    accuracy                           0.85      1334
   macro avg       0.88      0.85      0.84      1334
weighted avg       0.88      0.85      0.84      1334

Confusion Matrix
[[468 201]
 [  5 660]]
AUC:  0.8907133304112299
PAUC:  0.6234260420936694
Execution time: 39060.11 seconds
```

If you want to evaluate the MIMII dataset, then use the argument `--dataset mimii`. If you want to use CCC loss function, then use argument `--loss ccc`. Finally, there is an option to use reassigned spectrogram feature in addition to the melspectrogram. Use argument`--feature reassigned`. By default, loss history, distribution of errors, and confusion matrix are not shown. Use argument`--plot` to show these figures.

```bash
$ python baseline5.py --dataset mimii --loss ccc --feature reassigned

# Options:
  --dataset DATASET  Dataset to use for training and testing  {idmt, mimii}
  --feature FEATURE  Feature type to use for training and testing {mel, reassigned}
  --loss LOSS        Loss function to use for training the model {mse, ccc, mae, mape}
  --plot             Flag to plot the training loss (store true if flagged)
  --seed SEED        Seed number (default to 42)
```

## Results
Since I utilized GPU for training, the results is not reproducible. However, the results should be similar to the following if using CPU.
  
  ```bash
  # ./baseline5.py  # CPU
The error threshold is set to be:  107.05306549072266
              precision    recall  f1-score   support

      Normal       0.95      0.72      0.82       669
     Anomaly       0.78      0.96      0.86       665

    accuracy                           0.84      1334
   macro avg       0.86      0.84      0.84      1334
weighted avg       0.86      0.84      0.84      1334

Confusion Matrix
[[485 184]
 [ 27 638]]
AUC:  0.8304168492981331
PAUC:  0.553538081692312

  # ./run_mimii.sh # CPU
  The error threshold is set to be:  624.5870361328125
              precision    recall  f1-score   support

      Normal       0.84      0.78      0.81       138
     Anomaly       0.79      0.86      0.82       138

    accuracy                           0.82       276
   macro avg       0.82      0.82      0.81       276
weighted avg       0.82      0.82      0.81       276

Confusion Matrix
[[107  31]
 [ 20 118]]
AUC:  0.8997584541062801
PAUC:  0.8226268254126179
  ```

## Citation

```bibtex
B.T. Atmaja, 2024. "Evaluating Hyperparameter Optimization for Machinery Anomalous Sound Detection", In proc. TENCON 2024 Singapore (Accepted, TBA)
```

## References:  

1. <https://github.com/naveed88375/AI-ML/tree/master/Anomaly%20Detection%20in%20Industrial%20Equipment>
