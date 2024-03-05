import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_planes, out_planes * 2, kernel_size=3,
           stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block

    
def calculate_fid(generator, val_loader, batch_size=32):

    values = []
    fid = FrechetInceptionDistance(feature=64)

    with torch.no_grad():
        for images, gender_labels, age_labels in val_loader:
            age_labels = age_labels.unsqueeze(1)
            gender_labels = gender_labels.unsqueeze(1)
            age_condition = torch.zeros(batch_size, 8)
            age_condition.scatter_(1, age_labels, 1)
            condition = torch.cat([gender_labels, age_condition], dim=1).to('cuda')
            noise = torch.randn(images.size(0), 100).to('cuda')
            fake_images = generator(noise, condition)
            fake_feature = extract_features(fake_images).to('cpu')
            real_feature = extract_features(images)

            fid.update(real_feature.to(torch.uint8), real=True)
            fid.update(fake_feature.to(torch.uint8), real=False)
            values.append(fid.compute())
            fid.reset()
    
    return np.mean(values)

def extract_features(images):
    # Inception v3 모델에 이미지 전처리 및 특징 추출
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
    norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    features = norm(up(images))
    features = F.adaptive_avg_pool2d(features, output_size=64)
    features = F.sigmoid(features)
    return features

def calculate_is(generator, val_loader, batch_size=32):
    inception = InceptionScore(feature=64)
    values = []

    with torch.no_grad():
        for images, gender_labels, age_labels in val_loader:
            age_labels = age_labels.unsqueeze(1)
            gender_labels = gender_labels.unsqueeze(1)
            age_condition = torch.zeros(batch_size, 8)
            age_condition.scatter_(1, age_labels, 1)
            condition = torch.cat([gender_labels, age_condition], dim=1).to('cuda')
            noise = torch.randn(images.size(0), 100).to('cuda')
            fake_images = generator(noise, condition)
            fake_feature = extract_features(fake_images).to('cpu').to(torch.uint8)
            inception.update(fake_feature)
            values.append(inception.compute())
            inception.reset()

    return np.mean(values)