# Anomaly sound detection with CCC Loss function

## Downloading dataset

First, you need to download the IDMT and MIMII datasets. I located these data
inside `data` directory. If you locate them elsewhere, you need to adjust those paths (in `baseline.py`).

Link for download:  

- IDMT: <https://zenodo.org/record/7551261>
- MIMII Pump: <https://www.kaggle.com/datasets/senaca/mimii-pump-sound-dataset>

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
  --plot             Flag to plot the training loss
  --seed SEED        Seed number
```

## Results

## Citation

References:  

1. <https://github.com/naveed88375/AI-ML/tree/master/Anomaly%20Detection%20in%20Industrial%20Equipment>
