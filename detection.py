from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def detection(errs, test_labels, plot_cm=False):

    # Target labels
    target_labels = ["Normal", "Anomaly"]

    # Set threshold to be 10% more than the mean on normal sound signals
    # 669 is number of normal sound signals
    normal_len = 669
    thresh = np.mean(errs[:normal_len]) + 0.1*np.mean(errs[:normal_len])

    # Print the error threshold
    print("The error threshold is set to be: ", thresh)

    # Get prediction labels using error threshold
    pred_labels = np.array(errs) > thresh

    # Print the classification report
    print(classification_report(test_labels,
          pred_labels, target_names=target_labels))

    # print the confusion matrix
    print("Confusion Matrix")
    print(confusion_matrix(test_labels, pred_labels))

    # Plot the confusion matrix
    if plot_cm:
        cm = confusion_matrix(test_labels, pred_labels)
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
