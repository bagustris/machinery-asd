# Required packages
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
def autoencoder_baseline_mel(input_dims):
    # input layer
    inputLayer = Input(shape=(input_dims,))
    # Encoder block
    x = Dense(128, activation="relu")(inputLayer)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    # Latent space
    x = Dense(8, activation="relu")(x)
    # Decoder block
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    # Output layer
    x = Dense(input_dims, activation=None)(x)
    # Create and return the model
    return Model(inputs=inputLayer, outputs=x)


def autoencoder_baseline_reassigned(input_dims):
    # input layer
    inputLayer = Input(shape=(input_dims,))
    # Encoder block
    x = Dense(256, activation="relu")(inputLayer)
    x = Dense(64, activation="relu")(x)
    # Latent space
    x = Dense(16, activation="relu")(x)
    # Decoder block
    x = Dense(64, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    # Output layer
    x = Dense(input_dims, activation=None)(x)
    # Create and return the model
    return Model(inputs=inputLayer, outputs=x)
