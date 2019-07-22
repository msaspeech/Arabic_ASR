import librosa
import matplotlib.pyplot as plt
import librosa.display


# TODO : Add doc to this class
class AudioInput:
    def __init__(self, path, transcript):
        data, self.sample_rate = librosa.load(path)
        self.mfcc = self.extract_mfcc(data)
        self.audio_length = self.get_audio_length(data)
        self.audio_path = path
        self.audio_transcript = transcript

    def extract_mfcc(self, data):
        """
        Extract MFCC sequence from audio.
        :return: numpy.ndarray
                MFCC sequence
        """
        mfcc = librosa.feature.mfcc(y=data, sr=self.sample_rate, n_mfcc=40)
        return mfcc

    def get_audio_length(self, data):
        """
        Extract audio lenght in seconds
        :return: float
                audio length in seconds
        """
        return len(data) / self.sample_rate

    # TODO : Add the possibility to save a spectrogram
    def plot_audio_spectrogram(self):
        """
        Plot spectrogram
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(self.mfcc, x_axis='time')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def set_transcript(self, transcript):
        """
        Set transcript for the audio input
        """
        self.audio_transcript = transcript
