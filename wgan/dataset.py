import os
from torch.utils.data import Dataset, Subset, random_split
import torchvision.datasets as datasets

import pandas as pd
from PIL import Image
import random
from typing import Tuple
from natsort import natsorted
import gdown
import os
import zipfile

DATA_PATH = "/workspace/VoxCeleb/vox1"
FILE_NAME = "vox1_age_meta.csv"
CELEBA_PATH = 'data/celeba'
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

class CelebADataset(Dataset):
  def __init__(self, root_dir=CELEBA_PATH, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    # Root directory for the dataset
    data_root = 'data/celeba'
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'
    # URL for the CelebA dataset
    url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'

    # Create required directories 
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(dataset_folder)

        # Download the dataset from google drive
        gdown.download(url, download_path, quiet=False)

        # Unzip the downloaded file 
        with zipfile.ZipFile(download_path, 'r') as ziphandler:
            ziphandler.extractall(dataset_folder)
            
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img