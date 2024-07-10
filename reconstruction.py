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

    reconstruction_errors = []
    batch_size = 16
    batch_features = []
    batch_labels = []

    for eval_filename, label in zip(test_files, test_labels):
        signal, sr = load_sound_file(eval_filename)
        if feature == "mel":
            eval_features = extract_signal_features(signal, sr, n_mels=n_mels, frames=frames, n_fft=n_fft)
        elif feature == "reassigned":
            eval_features = extract_reassigned_freqs(signal, sr, frames=frames, n_fft=n_fft)
        else:
            raise ValueError("Invalid feature type. Choose 'mel' or 'reassigned'")

        batch_features.append(eval_features)
        batch_labels.append(label)

        if len(batch_features) == batch_size:
            predictions = model.predict(np.vstack(batch_features), batch_size=batch_size)
            for features, prediction in zip(batch_features, predictions):
                mse = np.mean(np.square(features - prediction))
                reconstruction_errors.append(mse)
            batch_features = []
            batch_labels = []

    # Process remaining features
    if batch_features:
        predictions = model.predict(np.vstack(batch_features), batch_size=len(batch_features))
        for features, prediction in zip(batch_features, predictions):
            mse = np.mean(np.square(features - prediction))
            reconstruction_errors.append(mse)

    # Plotting logic
    if plot:
        bin_width = 2
        bins = np.arange(min(reconstruction_errors), max(reconstruction_errors) + bin_width, bin_width)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist([reconstruction_errors[i] for i in range(len(reconstruction_errors)) if test_labels[i] == 0], bins=bins, alpha=0.5, color="b", label="Normal")
        ax.hist([reconstruction_errors[i] for i in range(len(reconstruction_errors)) if test_labels[i] == 1], bins=bins, alpha=0.5, color="r", label="Anomaly")
        ax.set_xlabel("Reconstruction error")
        ax.set_ylabel("# Samples")
        ax.set_title("Reconstruction error distribution on the testing set", fontsize=16)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return reconstruction_errors