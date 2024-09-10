
  

# 🔊 Voice2Face-modeling

  

<img  src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img  src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img  src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img  src="https://img.shields.io/badge/NCP-03C75A?style=for-the-badge&logo=Naver&logoColor=white"> <img  src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img  src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img  src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img  src="https://img.shields.io/badge/Huggingface-FFD21E?style=for-the-badge&logo=Huggingface&logoColor=white">



## Project Structure

```
wcgan-gp
  ┣ dataset.py
  ┣ inference.py
  ┣ inference_options.py
  ┣ model.py
  ┣ train.py
  ┣ train_options.py
  ┗ utils.py
```
## Usage

  
 - Wasserstein GAN: [paper](https://arxiv.org/pdf/1701.07875) | [github](https://github.com/martinarjovsky/WassersteinGAN)
 - Wasserstein GAN with Gradient Penalty: [paper](https://arxiv.org/abs/1704.00028) | [github](https://github.com/igul222/improved_wgan_training)
 - Conditional GAN: [paper](https://arxiv.org/abs/1411.1784) | [github](https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py)

 - `dataset.py`: celebA dataset과 voxcelebHQ dataset을 정의하기 위해 작성된 스크립트입니다.

  

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
