import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import PIL
import numpy as np
import random
from torch.utils.data.dataloader import default_collate

class OLKAVSDataset(Dataset):
    def __init__(self, csv_file, mfcc_transform=None, face_transform=None, mode='train'):
        self.data_frame = pd.read_csv(csv_file)
        self.mfcc_transform = self.set_transform('mfcc')
        self.face_transform = self.set_transform('face')
        self.nframe_range = (100, 150)
        self.mode = mode
        if self.mode == 'train':
            self.train_name_map = {}
        elif self.mode == 'val':
            self.val_name_map = {}
        elif self.mode == 'test':
            self.test_name_map = {}

    def __len__(self):
        return len(self.data_frame)
    
    def set_transform(self, type='face'):
        if type == 'face':  
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == "mfcc":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # MFCC 이미지 로드 및 변환
        mfcc_image_path = self.data_frame.iloc[idx, 0]
        try:
            with open(mfcc_image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    mfcc_image = image.convert('RGB')
                    mfcc_image = self.mfcc_transform(mfcc_image)
        except OSError as e:
            print(f"Error reading file {mfcc_image_path}: {e}")
        # 얼굴 이미지 로드 및 변환
        face_image_path = self.data_frame.iloc[idx, 1]
        try:
            with open(face_image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    face_image = image.convert('RGB')
                    face_image = self.face_transform(face_image)
        except OSError as e:
            print(f"Error reading file {face_image_path}: {e}")

        # 나머지 값 로드
        video_env = self.data_frame.iloc[idx, 2]
        audio_noise = self.data_frame.iloc[idx, 3]
        gender = self.data_frame.iloc[idx, 4]
        age = self.data_frame.iloc[idx, 5]
        specificity = self.data_frame.iloc[idx, 6]
        speakerID = self.data_frame.iloc[idx, 7]
        if self.mode == 'train':
            self.train_name_map[speakerID] = idx
        elif self.mode == 'val':
            self.val_name_map[speakerID] = idx
        elif self.mode == 'test':
            self.test_name_map[speakerID] = idx

        sample = {
            'mfcc_image': mfcc_image,
            'face_image': face_image,
            'video_env': video_env,
            'audio_noise': audio_noise,
            'gender': gender,
            'age': age,
            'specificity': specificity,
            'speakerID': speakerID,
            'mfcc_image_path' : mfcc_image_path,
            'face_image_path': face_image_path

        }
        labels = {
            'gender' : gender,
            'age' : age,
            'noise' : audio_noise,
        }
        return face_image, mfcc_image, speakerID, labels
    
    def get_all_faces_of_id(self, index):
        '''
        Given a id, return all the faces of him as a batch tensor, with shape
        (N, C, H, W)
        '''
        face_image, mfcc_image, speakerID, labels = self.__getitem__(index)
        return face_image
 
    def get_all_mel_segments_of_id(self,
                                   index,
                                   shuffle=False):
        '''
        Given a id, return all the speech segments of him as a batch tensor,
        with shape (N, C, L)
        '''
        face_image, mfcc_image, speakerID, labels = self.__getitem__(index)
        return mfcc_image

    def crop_or_pad(self, mfcc, out_frame):
        '''
        mfcc padding/cropping function to cooperate with collate_fn
        '''
        channel, freq, cur_frame = mfcc.shape
        if cur_frame >= out_frame:
            # Just crop
            start = np.random.randint(0, cur_frame-out_frame+1)
            mfcc = mfcc[..., start:start+out_frame]
        else:
            # Padding
            zero_padding = np.zeros((freq, out_frame-cur_frame))
            zero_padding = self.mfcc_transform(zero_padding)
            if len(zero_padding.shape) == 1:
                zero_padding = zero_padding.view([-1, 1])
            mfcc = torch.cat([mfcc, zero_padding], -1)

        return mfcc
       
    def collate_fn(self, batch):
        min_nframe, max_nframe = self.nframe_range
        assert min_nframe <= max_nframe
        np.random.seed()
        num_frame = np.random.randint(min_nframe, max_nframe+1)
        #start = np.random.randint(0, max_nframe-num_frame+1)
        #batch = [(item[0], item[1][..., start:start+num_frame], item[2])
        #         for item in batch]
        if self.mode == 'train':
            name_map = self.train_name_map
        elif self.mode == 'val':
            name_map = self.val_name_map
        elif self.mode == 'test':
            name_map = self.test_name_map
        
        batch = [(item[0],
                  self.crop_or_pad(item[1], num_frame),
                  name_map[item[2]], item[3]) for item in batch]
        return default_collate(batch)