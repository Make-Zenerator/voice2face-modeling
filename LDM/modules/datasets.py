"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

import random
import numpy as np
import argparse
import time
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import webrtcvad
try:
    import vad_ex
except:
    from utils import vad_ex


def get_dataset_function(loss_function_str: str):

    if loss_function_str == 'SegDataset':
        return SegDataset


class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, self.input_size)
        x = self.scaler(x)
        x = np.transpose(x, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            return x, y, filename

        elif self.mode in ['test']:
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"



def vad_process(path):
    # VAD Process
    if path.endswith('.wav'):
        audio, sample_rate = vad_ex.read_wave(path)
    elif path.endswith('.m4a'):
        audio, sample_rate = vad_ex.read_m4a(path)
    else:
        raise TypeError('Unsupported file type: {}'.format(path.split('.')[-1]))

    vad = webrtcvad.Vad(1)
    frames = vad_ex.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
    total_wav = b""
    for i, segment in enumerate(segments):
        total_wav += segment
    # Without writing, unpack total_wav into numpy [N,1] array
    # 16bit PCM 기준 dtype=np.int16
    wav_arr = np.frombuffer(total_wav, dtype=np.int16)
    #print("read audio data from byte string. np array of shape:" + \
    #    str(wav_arr.shape))
    return wav_arr, sample_rate

def wav_to_mel(path, nfilt=40):
    '''
    Output shape: (nfilt, length)
    '''
    wav_arr, sample_rate = vad_process(path)
    #print("sample_rate:", sample_rate)
    logmel_feats = logfbank(
        wav_arr,
        samplerate=sample_rate,
        nfilt=nfilt)
    #print("created logmel feats from audio data. np array of shape:" \
    #    + str(logmel_feats.shape))
    return np.transpose(logmel_feats)