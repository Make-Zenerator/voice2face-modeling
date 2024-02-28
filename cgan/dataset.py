import os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset, random_split

import torchvision
from torchvision.utils import save_image
from torchvision import transforms

import pandas as pd
from PIL import Image
import random
from typing import Tuple

DATA_PATH = "../data/VoxCeleb/vox1"
FILE_NAME = "vox1_age_meta.csv"

FILE_PATH = os.path.join(DATA_PATH, FILE_NAME)

class HQVoxceleb(Dataset):
    def __init__(self, transform=None):
        super(HQVoxceleb, self).__init__()
        self.data_path = DATA_PATH
        self.df = pd.read_csv(FILE_PATH, sep='\t', index_col=False)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        folder_path = os.path.join(self.data_path, "masked_faces", self.df["VGGFace1 ID"][index])
        image_name = random.choice(os.listdir(folder_path))
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        gender_label_str = self.df["Gender"][index]
        if gender_label_str == None:
            print(self.df["VGGFace1 ID"])
        age_label = int(self.df["age"][index])        # 문자열을 정수로 변환
        
        # 성별을 숫자로 매핑
        gender_label = 0 if gender_label_str == "m" else 1

        if self.transform:
            image = self.transform(image)
        return image, gender_label, age_label

    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * 0.2)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
