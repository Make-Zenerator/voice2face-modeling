import numpy as np
import librosa
import random
from tqdm import tqdm


def adjust(stft):
  if stft.shape[1] == 601:
    return stft
  else:
    return np.concatenate((stft,stft[:,0:601 - stft.shape[1]]),axis = 1)

def compute_speech2face_stft(folder_name, segment_duration=6.0, sr=16000):
    index = 0 
    num_of_records = 3
    stft_result = np.zeros((len(os.listdir(folder_name))*num_of_records, 527,601, 2), dtype=np.float32)

    for folder in tqdm(os.listdir(folder_name)):
        for i in range(num_of_records):
            
            path_ = os.path.join(folder_name, folder)
            file_path = os.path.join(path_, random.choice(os.listdir(path_)))
            wav_file, sr = librosa.load(file_path, sr=16000, duration=6.0, mono=True)
            
            stft_ = librosa.core.stft(
                    wav_file,
                    n_fft=512, 
                    hop_length=int(np.ceil(0.01 * sr)),
                    win_length = int(np.ceil(0.025 * sr)),
                    window='hann', 
                    center=True,
                    pad_mode='reflect'
                )
            
            stft = adjust(stft_)
            for j in range(stft.shape[0]):
                for k in range(stft.shape[1]):
                    stft_result[index, j, k, 0] = stft[j, k].real
                    stft_result[index, j, k, 1] = stft[j, k].imag

            index += 1
    spectrogram = np.sign(stft_result) * (np.abs(stft_result) ** 0.3)
    return spectrogram


# test 함수
import os
import matplotlib.pyplot as plt
# Magnitude와 Phase를 이미지로 저장
def save_magnitude_and_phase_images(magnitude, phase, magnitude_filename="mag", phase_filename="phase"):
    # Magnitude 이미지 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                             y_axis='log', x_axis='time')
    plt.title('STFT Magnitude')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(magnitude_filename)
    plt.close()

    # Phase 이미지 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(phase, y_axis='log', x_axis='time', cmap='twilight')
    plt.title('STFT Phase')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(phase_filename)
    plt.close()

# 메인 실행 함수
def main(wav_file):    
    # STFT 변환
    spectrogram = compute_speech2face_stft(wav_file)
    np.save("data/voice/wav_1.npy", spectrogram)
    # 이미지로 저장
    # save_magnitude_and_phase_images(spectrogram[0], spectrogram[1])
    
    
if __name__ == "__main__":
    wav_file = 'data/voice'
    main(wav_file)