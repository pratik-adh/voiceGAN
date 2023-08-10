import librosa
import numpy as np
import os
import utility.helper as helper

def convert_mp3_to_pitch_timbre_intonation(mp3_folder):

    # Create a list of MP3 files in the folder.
    mp3_files = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]

    # Create a list to store the pitch values.
    pitch_values = []

    # Create a list to store the timbre values.
    timbre_values = []

    # Create a list to store the intonation values.
    intonation_values = []

    # For each MP3 file in the list:
    for mp3_file in mp3_files:
        # Extract the pitch, timbre, and intonation of the MP3 file using the librosa library.
        pitch, timbre, intonation = librosa.feature.extract_pitch_timbre_intonation(mp3_file)

        # Add the pitch, timbre, and intonation values to the lists.
        pitch_values.append(pitch)
        timbre_values.append(timbre)
        intonation_values.append(intonation)

    # Combine the lists of pitch, timbre, and intonation values into a dictionary.
    pitch_timbre_intonation = {
        "pitch": pitch_values,
        "timbre": timbre_values,
        "intonation": intonation_values
    }

    # Save the dictionary to a file.
    with open("{}.npy".format(helper.get_foldername(mp3_folder)), "wb") as f:
        np.save(f, pitch_timbre_intonation)

