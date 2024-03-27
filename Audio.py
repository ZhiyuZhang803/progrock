import librosa
import constants
import numpy as np
import matplotlib.pyplot as plt


class Audio:
    def __init__(self):
        self.name = "Untitled"

    def set_audio(self, audio, i, sr=constants.sr):
        self.audio = audio
        self.sr = sr
        self.name = f"Segment {i}"

    def load_audio(self, path):
        self.audio, self.sr = librosa.load(path, sr=constants.sr)
        self.trim_silence()

    def trim_silence(self):
        self.audio, _ = librosa.effects.trim(self.audio)

    def get_spectrogram(self):
        self.spectro = librosa.feature.melspectrogram(
            y=self.audio, sr=self.sr, n_mels=constants.n_mels
        )

    def plot_spectrogram(self):
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(self.spectro, ref=np.max)
        img = librosa.display.specshow(
            S_dB, x_axis="time", y_axis="mel", sr=self.sr, fmax=8000, ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Mel-frequency spectrogram")
        plt.show()

    def get_chromagram(self):
        self.chroma = librosa.feature.chroma_stft(S=self.spectro, sr=self.sr)

    def plot_chromagram(self):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(
            self.chroma, y_axis="chroma", x_axis="time", ax=ax
        )
        fig.colorbar(img, ax=ax)
        ax.set(title="Chromagram")
        plt.show()

    def get_onset(self):
        self.onset = librosa.onset.onset_strength(y=self.audio, sr=self.sr)
        self.onset = librosa.util.normalize(self.onset)

    def plot_onset(self):
        fig, ax = plt.subplots()
        ax.plot(librosa.times_like(self.onset), self.onset)
        ax.set(title="Onset strength", xlabel="Time", ylabel="Normalized strength")
        plt.show()

    def get_MFCC(self):
        self.MFCC = librosa.feature.mfcc(
            y=self.audio,
            n_fft=constants.n_fft,
            hop_length=constants.hop_length,
            n_mfcc=constants.n_mfcc,
        )
        self.MFCC = librosa.util.normalize(self.MFCC)

    def plot_MFCC(self):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(self.MFCC, x_axis="time", ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set(title="MFCC")
        plt.show()

    def get_all_features(self):
        self.get_spectrogram()
        self.get_chromagram()
        self.get_onset()
        self.get_MFCC()

    def plot_all_features(self):
        fig, ax = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        S_dB = librosa.power_to_db(self.spectro, ref=np.max)
        img0 = librosa.display.specshow(
            S_dB, x_axis="time", y_axis="mel", sr=self.sr, fmax=8000, ax=ax[0]
        )
        fig.colorbar(img0, ax=ax[0], format="%+2.0f dB")
        ax[0].set(title="Mel-frequency spectrogram")

        img1 = librosa.display.specshow(
            self.chroma, y_axis="chroma", x_axis="time", ax=ax[1]
        )
        fig.colorbar(img1, ax=ax[1])
        ax[1].set(title="Chromagram")

        ax[2].plot(librosa.times_like(self.onset), self.onset)
        fig.colorbar(img0, ax=ax[2])
        ax[2].set(title="Onset strength", xlabel="Time", ylabel="Normalized strength")

        img3 = librosa.display.specshow(self.MFCC, x_axis="time", ax=ax[3])
        fig.colorbar(img3, ax=ax[3])
        ax[3].set(title="MFCC")

        plt.tight_layout()
        plt.savefig(f"output/{self.name}.pdf")
        plt.close()
