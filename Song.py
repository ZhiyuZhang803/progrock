import librosa
import logging

# Disable the logging from `librosa`
librosa_logger = logging.getLogger('numba')
librosa_logger.setLevel(logging.ERROR)


class Song:
    def __init__(self, path, label):
        self.path = path
        self.label = label

        self.load_song()

    def load_song(self):
        y, sr = librosa.load(self.path)
        self.wave_form = y
        self.sampling_rate = sr
