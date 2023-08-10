import os
import wave
import numpy as np
import librosa
import utility.helper as helper
import glob
from scipy.io import wavfile


def collect_training_dataset():

  # Create a new directory to store the training dataset.
  parent_directory = "train"
  child_directory_one = "wav"
  child_directory_two = "npy"
  helper.create_parent_and_child_directory(parent_directory, child_directory_one)

  # Record 10 audio samples from the target voice.
  # for i in range(10):
  #   print("Recording training audio {}...".format(i))
  #   record_audio.record_audio(i, seconds=10)
    
  # wav_dir = "{}/{}".format(parent_directory, child_directory_one)
  # npy_dir = "{}/{}".format(parent_directory, child_directory_two)

  # # Get a list of all the wav files in the directory.
  # wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))

  # # Initialize a list to store audio data
  # audio_data_list = []

  # # Load audio data from each WAV file and store in the list
  # for wav_file in wav_files:
  #   _, audio_data = wavfile.read(wav_file)
  #   audio_data_list.append(audio_data)

  # # Convert the list of audio data to a NumPy array
  # audio_data_array = np.array(audio_data_list)

  # # Save the NumPy array as a single .npy file
  # helper.create_parent_and_child_directory(parent_directory, child_directory_two)
  # np.save("{}/training_dataset.npy".format(npy_dir), audio_data_array)
  



