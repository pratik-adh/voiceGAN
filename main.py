import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras import layers, models
import glob
import librosa
import soundfile

# Compute spectrogram for an audio file
def compute_spectrogram(audio_data):
    stft = tf.signal.stft(audio_data, frame_length=2048, frame_step=512)
    spectrogram = tf.abs(stft)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

def generate_audio(spectrogram):
    audio = tf.signal.inverse_spectrogram(spectrogram)
    audio = tf.squeeze(audio, axis=-1)
    return audio
    
def load_audio(file_path):
  audio_data, sampling_rate = librosa.load(file_path, sr=None)
  return audio_data, sampling_rate

def save_audio(audio, filename):
  soundfile.write(filename, audio, sr=8000)

if __name__ == "__main__":

  # Collect the training dataset from the target voice.
  # training_data.collect_training_dataset()

  # Record the user's voice as test data.
  # testing_data.record_user_voice()

#   your_voice_data = np.load("test/npy/testing_dataset.npy")
#   celebrity_voice_data = np.load("train/npy/training_dataset.npy")
#   your_voice_data = your_voice_data.reshape(1, celebrity_voice_data.shape[1])
    epochs = 100
    batch_size = 32

    # Directory paths
    wav_dir_test = "test/wav"
    wav_dir_train = "train/wav" 

    # Load testing audio files and compute spectrograms
    testing_audio_files = glob.glob(os.path.join(wav_dir_test, "*.wav"))
    spectrograms_a = []

    for audio_file in testing_audio_files:
        audio_data, _ = load_audio(audio_file)
        spectrogram = compute_spectrogram(audio_data)
        spectrograms_a.append(spectrogram)

    # Load training audio files and compute spectrograms
    training_audio_files = glob.glob(os.path.join(wav_dir_train, "*.wav"))
    spectrograms_b = []

    for audio_file in training_audio_files:
        audio_data, _ = load_audio(audio_file)
        spectrogram = compute_spectrogram(audio_data)
        spectrograms_b.append(spectrogram)

    spectrograms_a = np.array(spectrograms_a)
    spectrograms_b = np.array(spectrograms_b)

    # Create the discriminator network
    generator = models.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(7 * 7 * 128, activation="relu"),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding="same", activation="sigmoid")
    ])

    discriminator = models.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])


    epochs = 100
    batch_size = 32

    discriminator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002)
    generator_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002)

    for epoch in range(epochs):
        for batch_i in range(0, len(spectrograms_a), batch_size):
            real_spectrograms = spectrograms_b[batch_i:batch_i + batch_size]  # Use spectrograms_b
            batch_a = spectrograms_a[batch_i:batch_i + batch_size]

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                fake_spectrograms = generator(batch_a, training=True)

                real_labels = tf.ones([batch_size, 1])
                fake_labels = tf.zeros([batch_size, 1])

                real_logits = discriminator(real_spectrograms, training=True)
                fake_logits = discriminator(fake_spectrograms, training=True)

                discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real_logits))
                discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake_logits))
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake

                generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=fake_logits))

            gradients_disc = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

            gradients_gen = gen_tape.gradient(generator_loss, generator.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Discriminator loss: {discriminator_loss.numpy()}")
        print(f"Generator loss: {generator_loss.numpy()}")
    
    generated_spectrogram = generator(spectrograms_a[0:1], training=False)
    generated_audio = generate_audio(generated_spectrogram.numpy())
    save_audio(generated_audio, "generated_audio.wav")

