# Anomaly sound detection with CCC Loss function

## Downloading dataset

First, you need to download IDMT and MIMII dataset. I located these data
inside `data`. Directory, if you locate elsewhere, you need to adjust those paths.

Link for download:  

- IDMT: <https://zenodo.org/record/7551261>
- MIMII Pump: <https://www.kaggle.com/datasets/senaca/mimii-pump-sound-dataset>

## IDMT dataset

IDMT works out of the box with default MSE loss. You only need to run `baseline4.py`.

```bash
$ python baseline4.py

The error threshold is set to be:  105.20362167358398
              precision    recall  f1-score   support

      Normal       0.71      0.71      0.71       669
     Anomaly       0.71      0.71      0.71       665

    accuracy                           0.71      1334
   macro avg       0.71      0.71      0.71      1334
weighted avg       0.71      0.71      0.71      1334

Confusion Matrix
[[475 194]
 [191 474]]
AUC:  0.7904199961787878
PAUC:  0.5313513900398861
Execution time: 46.87 seconds
```

I you want to evaluate MIMII dataset, then use argument `--dataset mimii`. If you want to use CCC loss function, then use argument `--loss ccc`. Finally there is an option to use reassigned spectrogram feature in addition to melspectrogram. Use argument`--feature reassigned`. By default loss history, ditribution of errors and confusion matrix are not shown. Use argument`--plot` to show these figures.

```bash
$ python baseline4.py --dataset mimii --loss ccc --feature reassigned

# Options:
  --dataset DATASET  Dataset to use for training and testing
  --feature FEATURE  Feature type to use for training and testing
  --loss LOSS        Loss function to use for training the model
  --plot             Flag to plot the training loss
  --seed SEED        Seed number
```

References:  

1. <https://github.com/naveed88375/AI-ML/tree/master/Anomaly%20Detection%20in%20Industrial%20Equipment>
