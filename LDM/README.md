
  

# 🔊 Voice2Face-modeling

  

<img  src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img  src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img  src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img  src="https://img.shields.io/badge/NCP-03C75A?style=for-the-badge&logo=Naver&logoColor=white"> <img  src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img  src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img  src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img  src="https://img.shields.io/badge/Huggingface-FFD21E?style=for-the-badge&logo=Huggingface&logoColor=white">



## Project Structure
```
code
┣ config
┃ ┣ predict_template.yaml
┃ ┣ train.yaml
┃ ┗ train_template.yaml
┣ models
┃ ┣ diffusion
┃ ┃ ┗ diffusion_models.py
┃ ┣ module
┃ ┃ ┣ attention.py
┃ ┃ ┣ blocks.py
┃ ┃ ┣ decoder.py
┃ ┃ ┣ ema.py
┃ ┃ ┣ encoder.py
┃ ┃ ┣ noise_scheduler.py
┃ ┃ ┣ voice_encoder.py
┃ ┃ ┗ vqvae.py
┃ ┗ utils.py
┣ modules
┃ ┣ datasets.py
┃ ┣ earlystoppers.py
┃ ┣ losses.py
┃ ┣ metrics.py
┃ ┣ optimizers.py
┃ ┣ recoders.py
┃ ┣ scalers.py
┃ ┣ schedulers.py
┃ ┣ trainer.py
┃ ┣ utils.py
┃ ┗ vad_ex.py
┣ LDM_LoRA.ipynb
┣ LDM_models.py
┣ LoRA_train.py
┣ custom_pipeline.py
┣ inference.py
┣ logger.ipynb
┣ logger.py
┣ predict.py
┣ requirements.txt
┣ sf2f_encoder.py
┗ train.py
```

## Usage

 - Latent Diffusion: [paper](https://arxiv.org/abs/2112.10752) | [github](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file)
 - Low-Rank Adaptation: [paper](https://arxiv.org/abs/2106.09685) | [github](https://github.com/microsoft/LoRA)
  

#### config
 - 매개 변수를 관리하기 위해 사용되는 스크립트를 모아둔 폴더입니다.
 - `train.yaml`: train에 사용되는 매개 변수들을 모아두어 수정 및 확인이 용이하게 한 스크립트입니다.

#### models
 - `diffusion/diffusion_models.py`: diffusion model의 구조를 정의하고 config file을 통해 전달받은 매개 변수를 기반으로 모델을 build하는 스크립트입니다.
 - `module/attention.py`: diffusion model에서 동작하는 attention 매커니즘을 구현한 스크립트입니다.
 - `module/blocks.py`: diffusion model을 구성할 때 사용되는 핵심 block들의 형태를 정의하는 스크립트입니다.
 - `module/decoder.py`: diffusion model에서 encoder에서 받아온 데이터를 복원하기 위한 decoder 구조를 정의하는 스크립트입니다.
 - `module/ema.py`: exponential moving average를 통해 noise를 줄여 성능을 향상하기 위해 사용되는 스크립트입니다.
 - `module/encoder.py`: diffusion model의 encoder로 데이터의 condition을 입력받을 수 있도록 정의된 스크립트입니다.
 - `module/voice_encoder.py`: 기존 CLIP encoder를 대체하여 음성 데이터(.wav) 혹은 이미지(mel_spectrogram, mfcc 등)을 입력받는 인코더 구조를 정의하는 스크립트입니다.
 - `module/vqvae.py`: diffusion model의 학습 속도 향상을 위해 vactor quantization을 사용하여 데이터의 크기를 줄여 계산량을 줄이고, 이를 원본 크기로 복구하기 위한 vae 구조를 정의하는 스크립트입니다.
 - 
#### modules 
 - `dataset.py`: 모델에 사용되는 dataset을 정의하는 스크립트입니다.
 - `earlystoppers.py`: overfitting을 방지하기 위해 사용되는 earlystopping 기법을 적용하기 위한 스크립트입니다.
 - `losses.py`: 모델 학습에 사용되는 loss funtion을 정의하는 스크립트입니다.
 - `metrics.py`: 모델 성능 평가를 위해 사용되는 metric(psnr, fid, mssim 등)을 정의하는 스크립트입니다.
 - `optimizers.py`: 모델 학습에 사용되는 optimizer(adam, adamw, adamp, RSMproop 등)을 정의하는 스크립트입니다.
 - `recoders.py`: 모델 학습 과정을 저장하기 위한 스크립트입니다.
 - `trainer.py`: train에서 사용될 정보들(optimizer, scheduler, scalers 등)을 모두 정의하고 학습 과정에 대해 세부 정의하는 스크립트입니다.
 - `vad_ex.py`: data의 형태를 통일하기 위해 음성 데이터를 이미지 데이터(mel_spectrogram, mfcc)로 변환하는 스크립트입니다.
 - 
#### total
 - `LDM_LoRA.ipynb`: LDM의 학습을 위해 사용된 LoRA 구조를 테스트하기 위해 작성된 스크립트입니다.
 - `LDM_models.py`: Pre-trained LDM을 정의하고 pipeline의 customizing하기 위해 작성된 스크립트입니다.
 - `LoRA_train.py`: LoRA 구조를 통해 LDM을 학습시키는 스크립트입니다.
 - `custom_pipeline.py`: Pre-trained 모델과 voice encoder를 결합하여 학습 pipeline을 구성하기 위해 작성된 스크립트입니다.
 - `sf2f_encoder.py`: Pre-trained sf2f voice encoder를 추출하여 사용하기 위해 작성된 스크립트입니다.
 - `train.py`: LDM 모델을 학습하기 위해 작성된 스크립트입니다.
  

## Getting Started

  
### Setting up Virtual Environment

  
1. Initialize and update the server

```

su -

source .bashrc

```

  

2. Create and Activate a virtual environment in the project directory

  

```

conda create -n env python=3.8

conda activate env

```

  

4. To deactivate and exit the virtual environment, simply run:

  

```

deactivate

```

  

### Install Requirements

  

To Install the necessary packages listed in `requirements.txt`, run the following command while your virtual environment is activated:

```

pip install -r requirements.txt

```

  
  
## Links
- [Naver BoostCamp 6th github](https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-08)
