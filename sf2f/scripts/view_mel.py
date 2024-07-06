import pickle
import matplotlib.pyplot as plt

# 예시로 저장된 pickle 파일 경로
pickle_file = '/home/mz01/voice2face-modeling/newmel/audio/audio/mel_audio_001_004.pickle'

# pickle 파일에서 데이터 로드
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# MEL spectrogram 데이터와 관련 정보 추출
mel_spec = data['LogMel_Features']
spkid = data['spkid']
wavid = data['wavid']

# MEL spectrogram 시각화
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='dB')
plt.title(f'MEL Spectrogram: Speaker {spkid}, File {wavid}')
plt.xlabel('Time')
plt.ylabel('Mel Filter Banks')
plt.tight_layout()

# 이미지 파일로 저장
output_file = f'mel_spectrogram_speaker_{spkid}_file_{wavid}.png'
plt.savefig(output_file)

plt.show()
