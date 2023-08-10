import os
import record_audio
import librosa
import numpy as np
import utility.helper as helper
from scipy.io import wavfile


def record_user_voice():

  # Create a new directory to store the training dataset.
  parent_directory = "test"
  child_directory_one = "wav"
  child_directory_two = "npy"
  helper.create_parent_and_child_directory(parent_directory, child_directory_one)

  print("Recording testing audio...")
  record_audio.record_audio(10)

  # test_data_dir = "{}/{}".format(parent_directory, child_directory_one)
  # npy_dir = "{}/{}".format(parent_directory, child_directory_two)
  
  # wav_file = "{}/audio_sample.wav".format(test_data_dir)

  # # Load audio data from the WAV file
  # _, audio_data = wavfile.read(wav_file)

  # # Save the npy audio clip to a file.
  # helper.create_parent_and_child_directory(parent_directory, child_directory_two)
  # np.save("{}/testing_dataset.npy".format(npy_dir), audio_data)
