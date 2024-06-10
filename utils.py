from tqdm import tqdm
import librosa
import numpy as np
import tensorflow as tf

def generate_dataset(files_list, dataset, feature="mel", n_mels=64, frames=5,
                     n_fft=1024, hop_length=512):
    # Function to generate training dataset
    if feature == "mel":
        dims = n_mels * frames
    elif feature == "reassigned":
        dims = 517 if dataset == 'idmt' else 626  # 513/517 is from error message
    else:
        raise ValueError("Invalid feature type. Choose 'mel' or 'reassigned'")
    
    dataset = None
    for index in tqdm(range(len(files_list)), desc="Extracting features"):
        # Load signal
        signal, sr = load_sound_file(files_list[index])

        if feature == "mel":
            # Extract melspectrogram from this signal:
            features = extract_signal_features(
                signal, sr, n_mels=n_mels, frames=frames, n_fft=n_fft
            )

        elif feature == "reassigned":
            features = extract_reassigned_freqs(
                signal, sr, frames=frames, n_fft=n_fft
            )

        if dataset is None:
            dataset = np.zeros(
                (features.shape[0] * len(files_list), dims), np.float32)

        dataset[features.shape[0] * index: features.shape[0] * (index + 1), :] = (
            features
        )

    return dataset


def extract_signal_features(signal, sr, n_mels=64, frames=5, n_fft=1024):
    # Compute a mel-scaled spectrogram:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )

    # Convert to decibel (log scale for amplitude):
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Generate an array of vectors as features for the current signal:
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1

    # Skips short signals:
    dims = frames * n_mels
    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)

    # Build N sliding windows (=frames) and concatenate them to build a feature vector:
    features = np.zeros((features_vector_size, dims), np.float32)
    for t in range(frames):
        features[:, n_mels * t: n_mels *
                 (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T

    return features


def load_sound_file(wav_name, mono=False, channel=0):
    # Load sound file
    signal, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
    # check if signal is multichannel, get first channel only
    if signal.ndim > 1:
        signal = signal[channel]
    return signal, sampling_rate


def ccc_loss(y_true, y_pred):
    """Calculate cordordance loss function"""
    # Mean of ground truth and predicted values
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)

    # Calculate covariance
    covariance = tf.reduce_mean(
        (y_true - y_true_mean) * (y_pred - y_pred_mean))

    # Calculate variances
    y_true_var = tf.math.reduce_variance(y_true)
    y_pred_var = tf.math.reduce_variance(y_pred)

    # Calculate CCC
    ccc = 2 * covariance / (y_true_var + y_pred_var +
                            (y_true_mean - y_pred_mean) ** 2)

    return 1 - ccc


def extract_reassigned_freqs(y, sr, frames=5, n_fft=1024):
    """extract reasiigned spetorgram, already in frames"""
    freqs, times, mags = librosa.reassigned_spectrogram(
        y=y, sr=sr, n_fft=1024)

    mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    # features_vector_size = mags_db.shape[1] - frames + 1

    # # skip short signals
    # dims = frames * mags_db.shape[0]
    # if features_vector_size < 1:
    #     return np.empty((0, dims), np.float32)
    
    # # Build N sliding windows can concatenate them to build feature vector
    # features = np.zeros((features_vector_size, dims), np.float32)
    # for t in range(frames):
    #     features[:,  mags_db.shape[0] * t:  mags_db.shape[0] * (t + 1)] = mags_db[:, t:t + features_vector_size].T

    return mags_db