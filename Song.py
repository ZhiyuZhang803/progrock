import os
import logging
import constants
import numpy as np
from Audio import Audio

# Disable the logging from `librosa`
librosa_logger = logging.getLogger("numba")
librosa_logger.setLevel(logging.ERROR)


class Song(Audio):
    def __init__(self, path, label):
        self.path = path
        self.name = os.path.basename(self.path)
        self.label = label

        super().load_audio(path)

    def split_song(self):
        samples_per_segment = constants.segment_length * self.sr
        total_segments = int(np.floor(len(self.audio) / samples_per_segment))

        self.segments = []
        for i in range(total_segments):
            segment = Audio()
            segment.set_audio(
                self.audio[i * samples_per_segment : (i + 1) * samples_per_segment], i
            )
            self.segments.append(segment)

    def process_song(self):
        self.split_song()
        for segment in self.segments:
            segment.get_all_features()

    def __str__(self):
        return f"(Song: {self.name}, Label: {self.label})"
