# Anomaly sound detection with IDMT dataset

## Downloading dataset

First, you need to download IDMT and MIMII dataset. You can place anywhere, and you path arguments to locate them.

## IDMT dataset

IDMT works out of the box. You only need to run `baseline_model.py`.

```bash
$ python baselin_model.py

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

References:  

1. <https://github.com/naveed88375/AI-ML/tree/master/Anomaly%20Detection%20in%20Industrial%20Equipment>
