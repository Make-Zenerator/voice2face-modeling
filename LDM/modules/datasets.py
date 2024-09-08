"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import torch
import pandas as pd
import random
import numpy as np
import argparse
import time
import torchvision.transforms as T
from _thread import start_new_thread
import queue
from python_speech_features import logfbank
import webrtcvad
from . import vad_ex

import json
import pickle
from .utils import imagenet_preprocess
import PIL
from torch.utils.data.dataloader import default_collate

def get_dataset_function(loss_function_str: str):

    if loss_function_str == 'OLKAVS':
        return OLKAVSDataset
    
OLKAVS_DIR = os.path.join('./data_OLKAVAS', 'OLKAVS')
class OLKAVSDataset(Dataset):
    def __init__(self,
                 data_dir=OLKAVS_DIR,
                 image_size=(256, 256),
                 face_type='masked',
                 image_normalize_method='imagenet',
                 mel_normalize_method='vox_mel',
                 nframe_range=(100, 150),
                 split_set='train',
                 split_csv=os.path.join(OLKAVS_DIR, 'annotation.csv'),
                 return_mel_segments=False,
                 mel_seg_window_stride=(125, 125),
                 image_left_right_avg=False,
                 image_random_hflip=False):
        '''
        A PyTorch Dataset for loading VoxCeleb 1 & 2 human speech
        (as mel spectrograms) and face (as image)

        Inputs:
        - data: Path to a directory where vox1 & vox2 data are held
        - face_type: 'masked' or 'origin'
        - image_size: Shape (h, w) to output the image
        - image_normalize_method: Method to normalize the image, 'imagenet' or
            'standard'
        - return_mel_segments: Return several segments of mel spectrogram
        - mel_seg_window_stride: Tuple (int, int), defines the window size and
            stride size when segmenting the mel spectrogram with sliding window
        - image_left_right_avg: flip the image, average the original and flipped
            image
        - image_random_hflip: Add random horizontal image flip to training set

        '''
        self.data_dir = data_dir
        self.image_size = image_size
        self.face_type = face_type
        self.face_dir = self.face_type + '_faces'
        self.image_normalize_method = image_normalize_method
        self.mel_normalize_method = mel_normalize_method
        self.nframe_range = nframe_range
        self.split_set = split_set
        self.split_csv = split_csv
        self.return_mel_segments = return_mel_segments
        self.mel_seg_window_stride = mel_seg_window_stride
        self.shuffle_mel_segments = True
        # This attribute is added to make the return segment mode to start from
        # a random time
        # Thus improve the randomness in fuser data
        self.mel_segments_rand_start = False
        self.image_left_right_avg = image_left_right_avg
        self.image_random_hflip = image_random_hflip

        self.load_split_dict()
        self.list_available_names()
        self.set_image_transform()
        self.set_mel_transform()

    def __len__(self):
        return len(self.available_names)

    def set_length(self, length):
        self.available_names = self.available_names[0:length]

    def __getitem__(self, index):
        '''
        Given an index, randomly return a face and a mel_spectrogram of this guy
        '''
        sub_dataset, name = self.available_names[index]
        # Face Image
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        image_jpg = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_jpg)

        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                # print("PIL Image:", np.array(image))
                image = image.convert('RGB')
                if self.image_left_right_avg:
                    arr = (np.array(image) / 2.0 + \
                        np.array(T.functional.hflip(image)) / 2.0).astype(
                            np.uint8)
                    image = PIL.Image.fromarray(arr, mode="RGB")
                image = self.image_transform(image)

        # Mel Spectrogram
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        mel_gram_pickle = random.choice(os.listdir(mel_gram_dir))
        mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
        if not self.return_mel_segments:
            # Return single segment
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
        else:
            log_mel = self.get_all_mel_segments_of_id(
                index, shuffle=self.shuffle_mel_segments)

        human_id = torch.tensor(index)

        return image, log_mel, human_id

    def get_all_faces_of_id(self, index):
        '''
        Given a id, return all the faces of him as a batch tensor, with shape
        (N, C, H, W)
        '''
        sub_dataset, name = self.available_names[index]
        faces = []
        # Face Image
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        for image_jpg in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_jpg)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    # print("PIL Image:", np.array(image))
                    image = self.image_transform(image.convert('RGB'))
                    faces.append(image)
        faces = torch.stack(faces)

        return faces

    def get_all_mel_segments_of_id(self,
                                   index,
                                   shuffle=False):
        '''
        Given a id, return all the speech segments of him as a batch tensor,
        with shape (N, C, L)
        '''
        sub_dataset, name = self.available_names[index]
        window_length, stride_length = self.mel_seg_window_stride
        segments = []
        # Mel Spectrogram
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        mel_gram_list = os.listdir(mel_gram_dir)
        if shuffle:
            random.shuffle(mel_gram_list)
        else:
            mel_gram_list.sort()
        seg_count = 0
        for mel_gram_pickle in mel_gram_list:
            mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
            mel_length = log_mel.shape[1]
            if self.mel_segments_rand_start:

                start = np.random.randint(mel_length - window_length) if mel_length > window_length else 0

                log_mel = log_mel[:, start:]
                mel_length = log_mel.shape[1]
            # Calulate the number of windows that can be generated
            num_window = 1 + (mel_length - window_length) // stride_length
            # Sliding Window
            for i in range(0, num_window):
                start_time = i * stride_length
                segment = log_mel[:, start_time:start_time + window_length]
                segments.append(segment)
                seg_count = seg_count + 1
                if seg_count == 20: # 20
                    segments = torch.stack(segments)
                    return segments
        segments = torch.stack(segments)
        return segments

    def set_image_transform(self):
        print('Dataloader: called set_image_size', self.image_size)
        image_transform = [T.Resize(self.image_size), T.ToTensor()]
        if self.image_random_hflip and self.split_set == 'train':
            image_transform = [T.RandomHorizontalFlip(p=0.5),] + \
                image_transform
        if self.image_normalize_method is not None:
            print('Dataloader: called image_normalize_method',
                self.image_normalize_method)
            image_transform.append(imagenet_preprocess(
                normalize_method=self.image_normalize_method))
        self.image_transform = T.Compose(image_transform)

    def set_mel_transform(self):
        mel_transform = [T.ToTensor(), ]
        print('Dataloader: called mel_normalize_method',
            self.mel_normalize_method)
        if self.mel_normalize_method is not None:
            mel_transform.append(imagenet_preprocess(
                normalize_method=self.mel_normalize_method))
        mel_transform.append(torch.squeeze)
        self.mel_transform = T.Compose(mel_transform)

    def load_split_dict(self):
        '''
        Load the train, val, test set information from split.json
        '''
        self.split_dict = pd.read_csv(self.split_csv)

    def list_available_names(self):
        '''
        Find the intersection of speech and face data
        '''
        self.available_names = []
        # List VoxCeleb1 data:
        for sub_dataset in (['voice']): #, 'vox2'
            mel_gram_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, 'mel_spectrograms'))
            face_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, self.face_dir))
            available = \
                set(mel_gram_available).intersection(face_available)
            for name in available:
                if name in self.split_dict[sub_dataset][self.split_set]:
                    self.available_names.append((sub_dataset, name))

        self.available_names.sort()

    def load_mel_gram(self, mel_pickle):
        '''
        Load a speech's mel spectrogram from pickle file.

        Format of the pickled data:
            LogMel_Features
            spkid
            clipid
            wavid

        Inputs:
        - mel_pickle: Path to the mel spectrogram to be loaded.
        '''
        # open a file, where you stored the pickled data
        file = open(mel_pickle, 'rb')
        # dump information to that file
        data = pickle.load(file)
        # close the file
        file.close()
        log_mel = data['LogMel_Features']
        #log_mel = np.transpose(log_mel, axes=None)

        return log_mel

    def crop_or_pad(self, log_mel, out_frame):
        '''
        Log_mel padding/cropping function to cooperate with collate_fn
        '''
        freq, cur_frame = log_mel.shape
        if cur_frame >= out_frame:
            # Just crop
            start = np.random.randint(0, cur_frame-out_frame+1)
            log_mel = log_mel[..., start:start+out_frame]
        else:
            # Padding
            zero_padding = np.zeros((freq, out_frame-cur_frame))
            zero_padding = self.mel_transform(zero_padding)
            if len(zero_padding.shape) == 1:
                zero_padding = zero_padding.view([-1, 1])
            log_mel = torch.cat([log_mel, zero_padding], -1)

        return log_mel

    def collate_fn(self, batch):
        min_nframe, max_nframe = self.nframe_range
        assert min_nframe <= max_nframe
        np.random.seed()
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        #start = np.random.randint(0, max_nframe-num_frame+1)
        #batch = [(item[0], item[1][..., start:start+num_frame], item[2])
        #         for item in batch]

        batch = [(item[0],
                  self.crop_or_pad(item[1], num_frame),
                  item[2]) for item in batch]
        return default_collate(batch)

    def count_faces(self):
        '''
        Count the number of faces in the dataset
        '''
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            # Face Image
            image_dir = os.path.join(
                self.data_dir, sub_dataset, self.face_dir, name)
            cur_count = len(os.listdir(image_dir))
            total_count = total_count + cur_count
        print('Number of faces in current dataset: {}'.format(total_count))
        return total_count

    def count_speech(self):
        '''
        Given a id, return all the speech segments of him as a batch tensor,
        with shape (N, C, L)
        '''
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            window_length, stride_length = self.mel_seg_window_stride
            # Mel Spectrogram
            mel_gram_dir = os.path.join(
                self.data_dir, sub_dataset, 'mel_spectrograms', name)
            mel_gram_list = os.listdir(mel_gram_dir)
            cur_count = len(mel_gram_list)
            total_count = total_count + cur_count
        print('Number of speech in current dataset: {}'.format(total_count))
        return total_count


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