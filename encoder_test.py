import os
import math
import torch
import encoder
import configure
import torchaudio

if __name__ == "__main__":
    voice_path = "data/voice"
    cfg = configure.Configure.make_configure()
    model = encoder.SpeechEncoder(cfg)
    voice = os.listdir(voice_path)
    print(voice[0])
    data_path = os.path.join(voice_path, voice[0])
    print(data_path)
    data, sr = torchaudio.load(data_path)
    win_length = math.ceil(sr/40)
    n_fft = 512
    hop_length = math.ceil(sr/100)
    spectrogram = torch.nn.Sequential(
        torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length
        ),
        torchaudio.transforms.AmplitudeToDB()
    )
    spec = spectrogram(data)
    print(spec.shape)