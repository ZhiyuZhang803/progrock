import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_and_segment_song(file_path, segment_length=10, sr_setting = 22050):
    """
    Load a song, trim silences, and segment into 10-second snippets.

    Parameters:
        file_path (str): The path to the audio file.
        segment_length (int): Length of each segment in seconds. Default is 10.

    Returns:
        segments (list): A list of audio segments.
    """
    # Load the song
    audio, sr = librosa.load(file_path, sr=sr_setting)

    # Trim the silence
    audio_trimmed, _ = librosa.effects.trim(audio)

    # Calculate the number of samples per segment
    samples_per_segment = segment_length * sr

    # Calculate the total number of segments needed (drop the snippets with less than 10s)
    total_segments = int(np.floor(len(audio_trimmed) / samples_per_segment))

    # Segment the audio
    segments = [audio_trimmed[i * samples_per_segment:(i + 1) * samples_per_segment] for i in range(total_segments)]

    return segments


def transfer_features(audio, sr=22050, n_mels=10, n_fft=2048, hop_length=512, n_chroma=12, n_mfcc=20):
    """
    Plot spectrogram, Mel-frequency spectrogram, chromagram, and onset strength of the audio.

    Parameters:
        audio (numpy.ndarray): The audio signal.
        sr (int): Sampling rate of the audio signal.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
        n_chroma (int): Number of chroma bins to produce.
        n_mfcc (int): Number of MFCCs to return.
    """
    # Compute Mel-frequency spectrogram
    M = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    M_DB = librosa.power_to_db(M, ref=np.max).tolist()

    # Compute chromagram
    C = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma).tolist()

    # Compute normalized onset strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_env = librosa.util.normalize(onset_env).tolist()

    # Compute MFCC
    MFCCs = librosa.feature.mfcc(y=audio, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    MFCCs_normalized = librosa.util.normalize(MFCCs).tolist()

    return M_DB, C, onset_env, MFCCs_normalized