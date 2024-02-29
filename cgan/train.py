import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import HQVoxceleb
from model import GenderDiscriminator, Discriminator, AgeDiscriminator, ConditionalGenrator
from utils import calculate_fid
from tqdm import tqdm
import wandb

def custom_collate_fn(batch):
    images, gender_labels, age_labels = zip(*batch)
    images = torch.stack(images, dim=0).float()  # 이미지를 torch.Tensor로 변환하고 float로 변환
    gender_labels = torch.tensor(gender_labels)
    age_labels = torch.tensor(age_labels)
    return images, gender_labels, age_labels

# 하이퍼파라미터 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
batch_size = 32
valid_batch_size = 32
epochs = 10000
noise_dim = 100
condition_dim = 9  # 성별 및 나이 조건에 따른 차원
image_size = (64, 64)
fid_threshold = 55  # FID 임계값

# 데이터 전처리 및 데이터로더 설정
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
dataset = HQVoxceleb(transform=transform)
train_set, val_set = dataset.split_dataset()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_set, batch_size=valid_batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

# 생성기 및 판별기 인스턴스 생성
generator = ConditionalGenrator(noise_dim=noise_dim, condition_dim=condition_dim).to(device)
discriminator = Discriminator().to(device)
# gender_discriminator = GenderDiscriminator().to(device)
# age_discriminator = AgeDiscriminator().to(device)

# 손실 함수 및 최적화 기준 설정
adversarial_loss = nn.BCEWithLogitsLoss()
optimizer_G = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizer_D_Gender = Adam(gender_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizer_D_Age = Adam(age_discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="Voice2Face",)
wandb.run.name = f"Conditional_GAN_{start_time}"

if not os.path.exists(f"output/{start_time}"): 
    os.makedirs(f"output/{start_time}") 


# 학습 루프
best_fid = float('inf')
for epoch in range(epochs):
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gender_labels, age_labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            images = images.to(device)
            batch_size = images.size(0)
            age_labels = age_labels.unsqueeze(1)
            gender_labels = gender_labels.unsqueeze(1)
            age_condition = torch.zeros(batch_size, 8)
            age_condition.scatter_(1, age_labels, 1)
            condition = torch.cat([gender_labels, age_condition], dim=1).to(device) 
            real_label = torch.full((batch_size,), 1.0, device=device)
            fake_label = torch.full((batch_size,), 0.0, device=device)
            
            # 생성기 훈련
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = generator(noise, condition)
            validity = discriminator(fake_images)
            g_loss = adversarial_loss(validity.view(-1), real_label)
            g_loss.backward()
            optimizer_G.step()
            
            # 판별기 훈련
            optimizer_D.zero_grad()
            real_images = images.to(device)
            real_validity = discriminator(real_images)
            real_loss = adversarial_loss(real_validity.view(-1), real_label)
            fake_validity = discriminator(fake_images.detach())
            fake_loss = adversarial_loss(fake_validity.view(-1), fake_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            wandb.log({
                "G_loss" : g_loss.item(),
                "D_loss" : d_loss.item()
            })
            tepoch.set_postfix(loss=g_loss.item())
            
        
        # 매 에폭 종료 후 FID 및 IS 평가
        fid = calculate_fid(generator, val_loader, valid_batch_size)
        wandb.log({"fid score" : fid})


        torch.save(generator.state_dict(), f"output/{start_time}/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"output/{start_time}/discriminator_{epoch}.pth")
        # FID 및 IS가 임계값 이내일 때 모델 저장
        if fid < best_fid:
            best_fid = fid
            torch.save(generator.state_dict(), f"output/{start_time}/generator_best.pth")
        
        # FID 또는 IS가 임계값을 벗어나면 학습 종료
        if fid > fid_threshold:
            print("FID 또는 IS가 임계값을 벗어나 학습 종료")
            break

torch.save(generator.state_dict(), f"output/{start_time}/generator_last.pth")
torch.save(discriminator.state_dict(), f"output/{start_time}/discriminator_last.pth")