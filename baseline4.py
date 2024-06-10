#!/usr/bin/env python

import argparse
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam

from detection import detection
from models import autoencoder_baseline_mel
from reconstruction import reconstruction
from utils import ccc_loss, generate_dataset

logger = logging.getLogger(__name__)


def load_idmt_dataset(
        normal_path,
        anomaly_path,
        test_path_normal,
        test_path_anomaly):

    anomaly_files = [os.path.join(anomaly_path, file)
                     for file in os.listdir(anomaly_path)]
    normal_files = [os.path.join(normal_path, file)
                    for file in os.listdir(normal_path)]

    train_files = normal_files + anomaly_files

    test_files_normal = [
        os.path.join(
            test_path_normal,
            file) for file in os.listdir(test_path_normal)]
    test_labels_normal = [0 for file in test_files_normal]
    test_files_abnormal = [
        os.path.join(
            test_path_anomaly,
            file) for file in os.listdir(test_path_anomaly)]
    test_labels_abnormal = [1 for file in test_files_abnormal]
    test_files = test_files_normal + test_files_abnormal
    test_labels = test_labels_normal + test_labels_abnormal
    test_labels = np.array(test_labels)

    return train_files, test_files, test_labels


def load_mimii_dataset(normal_path, anomaly_path):
    anomaly_files = [os.path.join(anomaly_path, file)
                     for file in os.listdir(anomaly_path)]
    normal_files = [os.path.join(normal_path, file)
                    for file in os.listdir(normal_path)]

    test_files = normal_files[-len(anomaly_files):] + anomaly_files
    test_labels = np.hstack(
        (np.zeros(
            len(anomaly_files)), np.ones(
            len(anomaly_files))))
    train_files = normal_files[:-len(anomaly_files)] + anomaly_files

    return train_files, test_files, test_labels


def main(dataset, feature, loss, plot, seed):
    log_dir = "log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=f'{log_dir}/{dataset}_{feature}_{loss}_{seed}.log',
        format='%(asctime)s %(message)s',
        level=logging.INFO)
    logger.info('==================Started==================')
    start_time = time.time()
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    dataset_paths = {
        "idmt": {
            "normal_path": './data/idmt/train_cut/engine1_good',
            "anomaly_path": './data/idmt/train_cut/engine2_broken',
            "test_path_normal": './data/idmt/test_cut/engine1_good',
            "test_path_anomaly": './data/idmt/test_cut/engine2_broken'
        },
        "mimii": {
            "normal_path": './data/mimii_pump/normal/',
            "anomaly_path": './data/mimii_pump/abnormal/'
        }
    }

    if dataset == "idmt":
        train_files, test_files, test_labels = load_idmt_dataset(
            **dataset_paths[dataset])
    elif dataset == "mimii":
        train_files, test_files, test_labels = load_mimii_dataset(
            **dataset_paths[dataset])
    else:
        raise ValueError("Invalid dataset")

    # Feature extraction and dataset generation
    n_mels = 64
    frames = 5
    n_fft = 1024
    hop_length = 512
    train_data = generate_dataset(
        train_files,
        dataset,
        feature,
        n_mels,
        frames,
        n_fft,
        hop_length)

    # Model design and compilation
    input_shape = train_data.shape[-1]

    loss_functions = {
        "mse": "mean_squared_error",
        "ccc": ccc_loss,
        "mae": "mean_absolute_error",
        "mape": "mean_absolute_percentage_error",
    }

    if loss not in loss_functions:
        raise ValueError("Invalid loss function")

    model_loss = loss_functions[loss]

    lr = 1e-3
    batch_size = 512
    epochs = 30

    if feature == "mel":
        baseline_model = autoencoder_baseline_mel(input_shape)
    elif feature == "reassigned":
        baseline_model = autoencoder_baseline_mel(input_shape)

    baseline_model.compile(loss=model_loss, optimizer=Adam(learning_rate=lr))
    baseline_model.summary()

    # Model training
    baseline_hist = baseline_model.fit(
        train_data,
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2)

    # log loss history
    logger.info(f"Training loss: {baseline_hist.history['loss']}")
    # Plot model loss
    if plot:
        plt.figure(figsize=(8, 5), dpi=100)
        plt.plot(range(epochs), baseline_hist.history["loss"])
        plt.xlabel("Epochs")
        plt.ylabel("mse loss")
        plt.title("Model training loss")
        plt.show()

    # Perform reconstruction using the test data files and calculate mse error
    print(f"Performing reconstruction on test data using feature: {feature}")
    reconstruction_errors = reconstruction(
        baseline_model,
        test_files,
        test_labels,
        feature,
        n_mels,
        frames,
        n_fft,
        plot)

    # Perform detection and evaluate model performance
    detection(reconstruction_errors, test_labels, dataset, plot)

    # Calculate AUC and pAUC
    auc = roc_auc_score(test_labels, reconstruction_errors)
    pauc = roc_auc_score(test_labels, reconstruction_errors, max_fpr=0.1)
    print("AUC: ", auc)
    print("PAUC: ", pauc)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    # save auc, pauc, execution time into log
    logger.info(f"AUC: {auc}")
    logger.info(f"PAUC: {pauc}")
    logger.info(f"Execution time: {execution_time:.2f} seconds")
    logger.info('==================Finished==================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline model for anomaly detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default="idmt",
        choices=[
            "idmt",
            "mimii"],
        help="Dataset to use for training and testing")
    parser.add_argument(
        "--feature",
        type=str,
        default="mel",
        choices=[
            "mel",
            "reassigned"],
        help="Feature type to use for training and testing")
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "ccc", "mae", "mape"],
        help="Loss function to use for training the model")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Flag to plot the training loss")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility")
    args = parser.parse_args()

    main(args.dataset, args.feature, args.loss, args.plot, args.seed)
