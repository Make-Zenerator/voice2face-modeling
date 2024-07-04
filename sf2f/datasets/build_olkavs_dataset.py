import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OLKAVSDataset(Dataset):
    def __init__(self, csv_file, mfcc_transform=None, face_transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.mfcc_transform = mfcc_transform
        self.face_transform = face_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # MFCC 이미지 로드 및 변환
        mfcc_image_path = self.data_frame.iloc[idx, 0]
        mfcc_image = Image.open(mfcc_image_path).convert('RGB')
        if self.mfcc_transform:
            mfcc_image = self.mfcc_transform(mfcc_image)
        
        # 얼굴 이미지 로드 및 변환
        face_image_path = self.data_frame.iloc[idx, 1]
        face_image = Image.open(face_image_path).convert('RGB')
        if self.face_transform:
            face_image = self.face_transform(face_image)

        # 나머지 값 로드
        video_env = self.data_frame.iloc[idx, 2]
        audio_noise = self.data_frame.iloc[idx, 3]
        gender = self.data_frame.iloc[idx, 4]
        age = self.data_frame.iloc[idx, 5]
        specificity = self.data_frame.iloc[idx, 6]
        speakerID = self.data_frame.iloc[idx, 7]

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
            'face_image_path':face_image_path

        }

        return sample

