import os
import pandas as pd
import numpy as np
# import cv2
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as preprocessing
from torchvision.transforms.functional import to_pil_image

data_root = "data/50k"
all_images = os.listdir(data_root)

number_of_samples = 2000
np.random.seed(seed=1)
sample_index = np.random.choice(len(all_images),number_of_samples)
sample_images = np.array(all_images)[sample_index]

PCA_data = []
for i in sample_images:
    this_image = list(np.asarray(Image.open(os.path.join(data_root, i))).flatten())
    PCA_data.append(this_image)
PCA_data = np.array(PCA_data)

pca = PCA(.1)
lower_dimensional_data = pca.fit_transform(PCA_data)
n_components = len(pca.explained_variance_ratio_)
approximation = pca.inverse_transform(lower_dimensional_data)

print(approximation.shape)

image = approximation.cpu().squeeze(0)
img = to_pil_image(image)
if not os.path.exists("pca"):
    os.mkdir("pca")
img.save(os.path.join("pca", 'pca_64.jpg'), "JPEG")

print(image.shape)
torch.save(approximation, "pca/pca_64.pkl")