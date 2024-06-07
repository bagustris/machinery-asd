# Import required packages
import numpy as np
from utils import extract_reassigned_freqs, extract_signal_features
from utils import load_sound_file
import matplotlib.pyplot as plt


def reconstruction(model, test_files, test_labels, feature, n_mels, frames, n_fft, plot):
    """
    Reconstructs the input features using a trained autoencoder model and calculates the reconstruction errors for the test set.

    Args:
        model (keras.Model): The trained autoencoder model.
        test_files (list): A list of file paths for the test audio files.
        test_labels (list): A list of labels (0 for normal, 1 for anomaly) for the test audio files.
        feature (str): The type of feature to use for extraction. Choose between 'mel' and 'reassigned'.
        n_mels (int): The number of mel-frequency bands to use for feature extraction.
        frames (int): The number of frames to use for feature extraction.
        n_fft (int): The number of FFT bins to use for feature extraction.

    Returns:
        list: A list of reconstruction errors for the test set.
    """

    # list to store reconstruction errors
    reconstruction_errors = []

    # Extract features from all test files in parallel
    eval_features_list = []
    for eval_filename in test_files:
        signal, sr = load_sound_file(eval_filename)
        if feature == "mel":
            eval_features = extract_signal_features(
                signal, sr, n_mels=n_mels, frames=frames, n_fft=n_fft
            )
        elif feature == "reassigned":
            eval_features = extract_reassigned_freqs(
                signal, sr, frames=frames, n_fft=n_fft
            )
        else:
            raise ValueError(
                "Invalid feature type. Choose 'mel' or 'reassigned'")
        eval_features_list.append(eval_features)

    # Get predictions from our autoencoder in batches
    batch_size = 32
    predictions = model.predict(
        np.vstack(eval_features_list), batch_size=batch_size)

    # Estimate the reconstruction errors
    for eval_features, prediction in zip(eval_features_list, predictions):
        mse = np.mean(np.mean(np.square(eval_features - prediction), axis=1))
        reconstruction_errors.append(mse)

    # Plot reconstruction errors for normal and anomaly signals
    data = np.column_stack(
        (range(len(reconstruction_errors)), reconstruction_errors))
    # Set bins
    bin_width = 0.25
    bins = np.arange(
        min(reconstruction_errors), max(
            reconstruction_errors) + bin_width, bin_width
    )

    # plot histogram of normal (test_labels == 0)
    # and anomaly signals (test_labels == 0) from data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data[test_labels == 0, 1], bins=bins,
            alpha=0.5, color="b", label="Normal")
    ax.hist(data[test_labels == 1, 1], bins=bins,
            alpha=0.5, color="r", label="Anomaly")

    # Label the plots
    if plot:
        ax.set_xlabel("Reconstruction error")
        ax.set_ylabel("# Samples")
        ax.set_title(
            "Reconstruction error distribution on the testing set", fontsize=16)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return reconstruction_errors
