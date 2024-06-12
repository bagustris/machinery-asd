#!/usr/bin/env python

import argparse
from gc import callbacks
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
from models import autoencoder_baseline_mel, autoencoder_baseline_reassigned
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


def objective(trial):
    """Objective function for Optuna optimization"""
    args.feature = trial.suggest_categorical("feature", ["mel", "reassigned"])
    args.lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    args.epochs = trial.suggest_int("epochs", 100, 1000)
    args.loss = trial.suggest_categorical("loss", ["mse", "ccc", "mae", "mape"])
    args.seed = trial.suggest_int("seed", 0, 100)
    args.normalize = trial.suggest_categorical("normalize", [True, False])
    args.patience = trial.suggest_int("patience", 5, 20) 
   
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

    # load dataset
    if args.dataset == "idmt":
        train_files, test_files, test_labels = load_idmt_dataset(
            **dataset_paths[args.dataset])
    elif args.dataset == "mimii":
        train_files, test_files, test_labels = load_mimii_dataset(
            **dataset_paths[args.dataset])
    else:
        raise ValueError("Invalid dataset")
    
    # value for feature extraction
    n_mels = 64
    frames = 5
    n_fft = 1024
    hop_length = 512

    # generate dataset
    train_data = generate_dataset(
        train_files,
        args.dataset,
        args.feature,
        n_mels,
        frames,
        n_fft,
        hop_length,
        args.normalize)
    
    # automatically get input shape for training
    input_shape = train_data.shape[-1]

    loss_functions = {
        "mse": "mean_squared_error",
        "ccc": ccc_loss,
        "mae": "mean_absolute_error",
        "mape": "mean_absolute_percentage_error",
    }

    if args.loss not in loss_functions:
        raise ValueError("Invalid loss function")
    
    model_loss = loss_functions[args.loss]

    if args.feature == "mel":
        baseline_model = autoencoder_baseline_mel(input_shape)
    elif args.feature == "reassigned":
        baseline_model = autoencoder_baseline_reassigned(input_shape)

    baseline_model.compile(loss=model_loss, optimizer=Adam(learning_rate=args.lr))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=args.patience, restore_best_weights=True)
 
    # Model training
    baseline_hist = baseline_model.fit(
        train_data,
        train_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[callback],
        verbose=2)
    
    # Perform reconstruction using the test data files and calculate mse error
    reconstruction_errors = reconstruction(
        baseline_model,
        test_files,
        test_labels,
        args.feature,
        n_mels,
        frames,
        n_fft,
        plot=False)
    
    # Perform detection and evaluate model performance
    auc = roc_auc_score(test_labels, reconstruction_errors)
    return auc


def main(
    dataset, feature, loss, plot, seed, normalize, optuna, learning_rate, batch_size, epochs
):
    # log_dir = f'./logs/{dataset}/{feature}/{loss}'
    log_dir = './logs/norm' if normalize else './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=f'{log_dir}/{dataset}_{feature}_{loss}_{seed}_{normalize}.log',
        format='%(asctime)s %(message)s',
        level=logging.INFO)
    logger.info('==================Started==================')
    # save arguments inside log
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Feature: {feature}")
    logger.info(f"Loss: {loss}")
    logger.info(f"Plot: {plot}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Normalize: {normalize}")

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
        hop_length, 
        normalize)

    # automatically get input shape for training
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

    lr = learning_rate
    batch_size = batch_size
    epochs = epochs

    # log hyperparameters
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    
    if feature == "mel":
        baseline_model = autoencoder_baseline_mel(input_shape)
    elif feature == "reassigned":
        baseline_model = autoencoder_baseline_reassigned(input_shape)

    baseline_model.compile(loss=model_loss, optimizer=Adam(learning_rate=lr))

    # log model summary
    baseline_model.summary(print_fn=lambda x: logger.info(x))
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, restore_best_weights=True)
    
    # log callback
    logger.info(f"Callback: {callback}")

    # Model training
    baseline_hist = baseline_model.fit(
        train_data,
        train_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[callback],
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

    # perform hyperparameter optimization using Optuna if flag is set
    if args.optuna:
        import optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        logger.info(f"Best trial: {study.best_trial}")
        logger.info(f"Best parameters: {study.best_params}")

        for key, value in study.best_params.items():
            setattr(args, key, value)
    print("Optimization finished")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline model for anomaly detection")
    parser.add_argument(
        "--dataset",
        type=str,
        default="idmt",
        choices=["idmt", "mimii"],
        help="Dataset to use for training and testing",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="mel",
        choices=["mel", "reassigned"],
        help="Feature type to use for training and testing",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "ccc", "mae", "mape"],
        help="Loss function to use for training the model",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Flag to plot the training loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize the features"
    )
    # add optuna argument to optimize hyperparameters
    parser.add_argument(
        "--optuna", action="store_true", help="Use optuna to optimize hyperparameters"
    )
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bands")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames")
    parser.add_argument("--n_fft", type=int, default=1024, help="Number of FFT points")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for training"
    )

    args = parser.parse_args()

    # print(args)
    print(f"Dataset: {args.dataset}")
    print(f"Feature: {args.feature}")
    print(f"Loss: {args.loss}")
    print(f"Plot: {args.plot}")
    print(f"Seed: {args.seed}")
    print(f"Normalize: {args.normalize}")
    print(f"Optuna: {args.optuna}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    main(
        args.dataset,
        args.feature,
        args.loss,
        args.plot,
        args.seed,
        args.normalize,
        args.optuna,
        args.lr,
        args.batch_size,
        args.epochs
    )
