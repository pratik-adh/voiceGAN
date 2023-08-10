import pyaudio
import wave
import os

def record_audio(i=0, seconds=10):

  chunk = 1024  # Record in chunks of 1024 samples
  sample_format = pyaudio.paInt16  # 16 bits per sample
  channels = 1
  fs = 44100  # Record at 44100 samples per second
  filename = "train/wav/audio_sample_{}.wav".format(i) if i < 10 else "test/wav/audio_sample.wav"

  print(filename)
  # Create a new PyAudio object.
  pyaudio_object = pyaudio.PyAudio()

  # Open a new audio stream.
  audio_stream = pyaudio_object.open(
      format=sample_format,
      channels=channels,
      rate=fs,
      frames_per_buffer=chunk,
      input=True)

  # Start recording audio.
  audio_data = []

  for _ in range(int(seconds * fs / chunk)):
    raw_audio_data = audio_stream.read(chunk)
    audio_data.append(raw_audio_data)
  
    # Check if the audio data is long enough.
    if len(audio_data) >= seconds * 44100:
      break

  # Stop recording audio and close the audio stream.
  audio_stream.stop_stream()
  audio_stream.close()


  # Save the recorded data as a WAV file
  wf = wave.open(filename, 'wb')
  wf.setnchannels(channels)
  wf.setsampwidth(pyaudio_object.get_sample_size(sample_format))
  wf.setframerate(fs)
  wf.writeframes(b''.join(audio_data))
  wf.close()

  # Close the PyAudio object.
  pyaudio_object.terminate()
