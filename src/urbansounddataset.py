from torch.utils.data import Dataset
import pandas as pd
import torch
import torchaudio
import soundfile
import os

torchaudio.set_audio_backend("soundfile")

class UrbanSoundDataset(Dataset):

    # Creating the constructor
    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation, 
                 target_sample_rate, 
                 num_samples):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples


    def __len__(self):
        return len(self.annotations)

    #len(usd)

    # When we call this, we will get the waveform and its label
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000) Turning stereo to mono

        # Resampling the loaded audio to make sure everything is uniform
        signal = self._resample_if_necessary(signal, sr)

        # We want to turn the audio samples into mono if needed
        signal = self._mix_down_if_necessary(signal)

        # Performing Mel Spectrogram transformation here
        signal = self.transformation(signal)

        return signal, label
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        # A signal can have multiple channels, we want to mix everything down to a single channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "./data/UrbanSound8K/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    # Creating a Mel Spectrogram Transformation
    # Documentation for torchaudio.transforms.MelSpectrogram: https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE, 
                            NUM_SAMPLES)

    print(f"Number of samples: {len(usd)}")

    signal, label = usd[0]