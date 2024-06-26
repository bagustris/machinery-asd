#!/usr/bin/env python3

# Import required packages
from ast import main
from sklearn.metrics import roc_auc_score
import os
import librosa
import numpy as np
from tqdm import tqdm
import seaborn as sns
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from models import autoencoder_baseline
from tensorflow.keras.optimizers import Adam
from utils import generate_dataset, load_sound_file
from reconstruction import reconstruction
from detection import detection
from pathlib import Path
from utils import ccc_loss


def main(normal_path, anomaly_path,
         test_path_normal, test_path_anomaly, feature, loss, plot_cm=False):
    anomaly_files = [os.path.join(anomaly_path, file)
                     for file in os.listdir(anomaly_path)]
    normal_files = [os.path.join(normal_path, file)
                    for file in os.listdir(normal_path)]

    train_files = normal_files + anomaly_files

    test_files_normal = [os.path.join(test_path_normal, file)
                         for file in os.listdir(test_path_normal)]
    test_labels_normal = [0 for file in test_files_normal]
    test_files_abnormal = [os.path.join(test_path_anomaly, file)
                           for file in os.listdir(test_path_anomaly)]

    test_labels_abnormal = [1 for file in test_files_abnormal]
    test_files = test_files_normal + test_files_abnormal
    test_labels = test_labels_normal + test_labels_abnormal
    test_labels = np.array(test_labels)

    # Feature extraction and dataset generation
    n_mels = 64
    frames = 5
    n_fft = 1024
    hop_length = 512
    train_data = generate_dataset(
        train_files, n_mels, frames, n_fft, hop_length)

    # Shape of the input data
    input_shape = n_mels*frames
    # Loss function
    if loss == 'mse':
        model_loss = 'mean_squared_error'
    elif loss == 'ccc':
        model_loss = ccc_loss
    else:
        raise ValueError('Invalid loss function')

    # Optimizer learning rate
    lr = 1e-3
    # Batch size and number of epochs to train the model
    batch_size = 512
    epochs = 30
    # Create the baseline model and compile it with the hyperparameters
    baseline_model = autoencoder_baseline(input_shape)
    baseline_model.compile(loss=model_loss,
                           optimizer=Adam(learning_rate=lr))
    # Print model summary
    baseline_model.summary()

    # Model training
    baseline_hist = baseline_model.fit(
        train_data,
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2
    )

    # Plot model loss
    plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(range(epochs), baseline_hist.history['loss'])
    # Label the plot
    plt.xlabel('Epochs')
    plt.ylabel('mse loss')
    plt.title('Model training loss')
    plt.show()

    # Perform reconstruction using the test data files and calculate mse error scores
    reconstruction_errors = reconstruction(baseline_model,
                                           test_files,
                                           test_labels,
                                           n_mels,
                                           frames,
                                           n_fft)

    # Perform detection and evaluate model performance
    detection(reconstruction_errors, test_labels)

    # calculate auc and pauc
    auc = roc_auc_score(test_labels, reconstruction_errors)
    pauc = roc_auc_score(test_labels, reconstruction_errors, max_fpr=0.1)
    print('AUC: ', auc)
    print('PAUC: ', pauc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_train_path", type=Path,
                        default='/home/bagus/data/IDMT/train_cut/engine1_good')
    parser.add_argument("--anomaly_train_path", type=Path,
                        default='/home/bagus/data/IDMT/train_cut/engine2_broken')
    parser.add_argument("--normal_test_path", type=Path,
                        default='/home/bagus/data/IDMT/test_cut/engine1_good')
    parser.add_argument("--anomaly_test_path", type=Path,
                        default='/home/bagus/data/IDMT/test_cut/engine2_broken')
    parser.add_argument("--feature", type=str,
                        default='mel', choices=['mel', 'ifgram'])
    parser.add_argument("--loss", type=str,
                        default='mse', choices=['mse', 'ccc'])
    parser.add_argument("--plot_cm", type=bool,
                        default=False)
    args = parser.parse_args()
    # print args information
    print("Normal Train Path: ", args.normal_train_path)
    print("Anomaly Train Path: ", args.anomaly_train_path)
    print("Normal Test Path: ", args.normal_test_path)
    print("Anomaly Test Path: ", args.anomaly_test_path)
    print("Feature: ", args.feature)
    print("Loss: ", args.loss)
    print("Plot Confusion Matrix: ", args.plot_cm)
    main(args.normal_train_path, args.anomaly_test_path, args.normal_test_path,
         args.anomaly_test_path, args.feature, args.loss, args.plot_cm)
