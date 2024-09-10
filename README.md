
  

# 🔊 Voice2Face-modeling

  

<img  src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img  src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"> <img  src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img  src="https://img.shields.io/badge/NCP-03C75A?style=for-the-badge&logo=Naver&logoColor=white"> <img  src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=Linux&logoColor=white"> <img  src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white"> <img  src="https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=Pytorch&logoColor=white"> <img  src="https://img.shields.io/badge/Huggingface-FFD21E?style=for-the-badge&logo=Huggingface&logoColor=white">



## Project Structure

```
code
┣ LDM
┃ ┣ config
┃ ┃ ┣ predict_template.yaml
┃ ┃ ┣ train.yaml
┃ ┃ ┗ train_template.yaml
┃ ┣ models
┃ ┃ ┣ diffusion
┃ ┃ ┃ ┗ diffusion_models.py
┃ ┃ ┣ module
┃ ┃ ┃ ┣ attention.py
┃ ┃ ┃ ┣ blocks.py
┃ ┃ ┃ ┣ decoder.py
┃ ┃ ┃ ┣ ema.py
┃ ┃ ┃ ┣ encoder.py
┃ ┃ ┃ ┣ noise_scheduler.py
┃ ┃ ┃ ┣ voice_encoder.py
┃ ┃ ┃ ┗ vqvae.py
┃ ┃ ┗ utils.py
┃ ┣ modules
┃ ┃ ┣ datasets.py
┃ ┃ ┣ earlystoppers.py
┃ ┃ ┣ losses.py
┃ ┃ ┣ metrics.py
┃ ┃ ┣ optimizers.py
┃ ┃ ┣ recoders.py
┃ ┃ ┣ scalers.py
┃ ┃ ┣ schedulers.py
┃ ┃ ┣ trainer.py
┃ ┃ ┣ utils.py
┃ ┃ ┗ vad_ex.py
┃ ┣ LDM_LoRA.ipynb
┃ ┣ LDM_models.py
┃ ┣ LoRA_train.py
┃ ┣ custom_pipeline.py
┃ ┣ inference.py
┃ ┣ logger.ipynb
┃ ┣ logger.py
┃ ┣ predict.py
┃ ┣ requirements.txt
┃ ┣ sf2f_encoder.py
┃ ┗ train.py
┣ SimSwap
┃ ┣ docs
┃ ┃ ┣ ...
┃ ┣ insightface_func
┃ ┃ ┣ utils
┃ ┃ ┃ ┗ face_align_ffhqandnewarc.py
┃ ┃ ┣ __init__.py
┃ ┃ ┣ face_detect_crop_multi.py
┃ ┃ ┗ face_detect_crop_single.py
┃ ┣ models
┃ ┃ ┣ __init__.py
┃ ┃ ┣ arcface_models.py
┃ ┃ ┣ base_model.py
┃ ┃ ┣ config.py
┃ ┃ ┣ fs_model.py
┃ ┃ ┣ fs_networks.py
┃ ┃ ┣ fs_networks_512.py
┃ ┃ ┣ fs_networks_fix.py
┃ ┃ ┣ models.py
┃ ┃ ┣ networks.py
┃ ┃ ┣ pix2pixHD_model.py
┃ ┃ ┣ projected_model.py
┃ ┃ ┣ projectionhead.py
┃ ┃ ┗ ui_model.py
┃ ┣ options
┃ ┃ ┣ base_options.py
┃ ┃ ┣ test_options.py
┃ ┃ ┗ train_options.py
┃ ┣ parsing_model
┃ ┃ ┣ model.py
┃ ┃ ┗ resnet.py
┃ ┣ pg_modules
┃ ┃ ┣ blocks.py
┃ ┃ ┣ diffaug.py
┃ ┃ ┣ projected_discriminator.py
┃ ┃ ┗ projector.py
┃ ┣ simswaplogo
┃ ┃ ┗ ...
┃ ┣ util
┃ ┃ ┣ add_watermark.py
┃ ┃ ┣ gifswap.py
┃ ┃ ┣ html.py
┃ ┃ ┣ image_pool.py
┃ ┃ ┣ json_config.py
┃ ┃ ┣ logo_class.py
┃ ┃ ┣ norm.py
┃ ┃ ┣ plot.py
┃ ┃ ┣ reverse2original.py
┃ ┃ ┣ save_heatmap.py
┃ ┃ ┣ util.py
┃ ┃ ┣ videoswap.py
┃ ┃ ┣ videoswap_multispecific.py
┃ ┃ ┣ videoswap_specific.py
┃ ┃ ┗ visualizere.py
┃ ┣ cog.yaml
┃ ┣ download-weights.sh
┃ ┣ inference_swap.py
┃ ┣ predict.py
┃ ┣ test_one_image.py
┃ ┣ test_video_swap_multispecific.py
┃ ┣ test_video_swapmulti.py
┃ ┣ test_video_swapsingle.py
┃ ┣ test_video_swapspecific.py
┃ ┣ test_wholeimage_swap_multispecific.py
┃ ┣ test_whileimage_swapmulti.py
┃ ┣ test_wholeimage_swapsingle.py
┃ ┣ test_wholeimage_swapspecific.py
┃ ┗ train.py
┣ pytorch_template
┃ ┗ ...
┣ sf2f
┃ ┣ datasets
┃ ┃ ┣ __init__.py
┃ ┃ ┣ build_dataset.py
┃ ┃ ┣ build_olkavs_dataset.py
┃ ┃ ┣ utils.py
┃ ┃ ┗ vox_dataset.py
┃ ┣ images
┃ ┃ ┗ ...
┃ ┣ models
┃ ┃ ┣ __init__.py
┃ ┃ ┣ attention.py
┃ ┃ ┣ crn.py
┃ ┃ ┣ discriminators.py
┃ ┃ ┣ encoder_decoder.py
┃ ┃ ┣ face_decoders.py
┃ ┃ ┣ fusers.py
┃ ┃ ┣ inception_resnet_v1.py
┃ ┃ ┣ layers.py
┃ ┃ ┣ model_collection.py
┃ ┃ ┣ model_setup.py
┃ ┃ ┣ networks.py
┃ ┃ ┣ perceptual.py
┃ ┃ ┗ voice_encoders.py
┃ ┣ options
┃ ┃ ┣ data_opts
┃ ┃ ┃ ┣ olk.yaml
┃ ┃ ┃ ┗ vox.yaml
┃ ┃ ┣ vox
┃ ┃ ┃ ┣ baseline
┃ ┃ ┃ ┃ ┗ v2f.yaml
┃ ┃ ┃ ┗ sf2f
┃ ┃ ┃ ┃ ┣ olkavs_sf2f_1st_stage.yaml
┃ ┃ ┃ ┃ ┣ sf2f_1st_stage.yaml
┃ ┃ ┃ ┃ ┣ sf2f_fuser.yaml
┃ ┃ ┃ ┃ ┣ sf2f_mid_1st_stage.yaml
┃ ┃ ┃ ┃ ┗ sf2f_mid_fuser.yaml
┃ ┃ ┣ __init__.py
┃ ┃ ┗ opts.py
┃ ┣ scripts
┃ ┃ ┣ build_demo_set.py
┃ ┃ ┣ compute_diversity_score.py
┃ ┃ ┣ compute_fid_score.py
┃ ┃ ┣ compute_inception_score.py
┃ ┃ ┣ compute_mel_mean_var.py
┃ ┃ ┣ compute_vggface_score.py
┃ ┃ ┣ convert_wav_to_mel.py
┃ ┃ ┣ create_split_json.py
┃ ┃ ┣ download_vggface_weights.sh
┃ ┃ ┣ install_requirements.py
┃ ┃ ┣ print_args.py
┃ ┃ ┣ sample_mel_spectrograms.py
┃ ┃ ┣ strip_checkpoint.py
┃ ┃ ┣ strip_old_args.py
┃ ┃ ┗ watch_data.py
┃ ┣ utils
┃ ┃ ┣ visualization
┃ ┃ ┃ ┣ __init__.py
┃ ┃ ┃ ┣ html.py
┃ ┃ ┃ ┣ plot.py
┃ ┃ ┃ ┣ tsne.py
┃ ┃ ┃ ┗ vis.py
┃ ┃ ┣ __init__.py
┃ ┃ ┣ bilinear.py
┃ ┃ ┣ box_utils.py
┃ ┃ ┣ common.py
┃ ┃ ┣ compute_metrics.py
┃ ┃ ┣ connect_mlflow.py
┃ ┃ ┣ evaluate.py
┃ ┃ ┣ evaluate_fid.py
┃ ┃ ┣ filter_pickle.py
┃ ┃ ┣ logger.py
┃ ┃ ┣ losses.py
┃ ┃ ┣ metrics.py
┃ ┃ ┣ pgan_utils.py
┃ ┃ ┣ s2f_evaluator.py
┃ ┃ ┣ training_utils.py
┃ ┃ ┣ utils.py
┃ ┃ ┣ vad_ex.py
┃ ┃ ┣ wandb_logger.py
┃ ┃ ┗ wav2mel.py
┃ ┣ infer.py
┃ ┗ inference_fuser.py
┃ ┣ requirements.txt
┃ ┣ test.py
┃ ┗ train.py.py
┗ wcgan-gp
  ┣ dataset.py
  ┣ inference.py
  ┣ inference_options.py
  ┣ model.py
  ┣ train.py
  ┣ train_options.py
  ┗ utils.py
┗ README.md
┗ requirements.txt
┗ train.sh
┗ voxceleb_download.sh
...
```
## Usage

  

#### LDM
 - Latent Diffusion: [paper](https://arxiv.org/abs/2112.10752) | [github](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file)
 - Low-Rank Adaptation: [paper](https://arxiv.org/abs/2106.09685) | [github](https://github.com/microsoft/LoRA)
 - 기존 Speech Fusion to Face 모델의 voice encoder를 활용하여 결과 이미지의 품질 향상을 위해 Latent Diffusion model을 구현한 폴더입니다.
 - 또한, Diffusion model의 원활한 학습을 위해 LoRA 구조를 추가하였고, 이를 통해 학습 시간 및 성능을 개선하였습니다.

#### SimSwap
 - SimSwap: [paper](https://arxiv.org/pdf/2106.06340v1) | [github](https://github.com/neuralchen/SimSwap)
 - 음성 데이터로 부터 생성된 얼굴을 기존 영상에 합성하기 위해 사용된 모델입니다.
 - 생성된 정면 얼굴을 영상 속 다양한 각도에 맞게 합성하기 위해 합성 속도보다 정확도와 품질이 보다 높은 모델을 선택하였습니다.
 - 합성이 완료된 영상을 gif 혹은 mp4 형태로 생성하여 출력합니다.

#### pytorch_template 
 - pytorch template: 참고[github](https://github.com/victoresque/pytorch-template)
 - 모델 개발의 효율성과 일관성을 유지하기 위해 사용한 형식입니다.
 - 개발한 모델을 팀원들이 이해하기 쉽도록 정리하여 공유하였습니다.

#### sf2f
 - Speech Fusion to Face: [paper](https://arxiv.org/abs/2006.05888) | [github](https://github.com/BAI-Yeqi/SF2F_PyTorch) | [page](https://sf2f.github.io/)
 - 음성 데이터 (.wav) 파일을 mel_spectrogram으로 변환한 후, 이를 통해 얼굴을 재생성하는 모델입니다.
 - `scripts/convert_wav_to_mel.py`: 음성 데이터(.wav) 파일을 일정한 크기(100x150)의 mel_spectrogram으로 변환하는 전처리를 수행하고 이를 pickle 파일로 저장하는 스크립트입니다.
 - `options/data_opts` : 데이터 셋을 생성할 때 사용하는 매개 변수들을 지정하는 스크립트들을 저장해둔 폴더로, vox celeb dataset과 olkavs dataset에 대한 스크립트가 사용됩니다.
 - `options/sf2f`: train과 inference 시에 사용되는 모든 매개 변수들을 지정하는 스크립트들을 저장해둔 폴더로, sf2f with vox와 sf2f with olkavs로 나뉘어져 있고, sf2f는 모델의 방식과 이미지 데이터의 크기에 따라 분류되어 있습니다.
 - `utils/compute_metrics.py`: 모델 성능 평가를 위해 사용되는 metrics를 선언하고 계산하는 것을 통해 모델 학습의 평가 지표로 사용하기 위한 스크립트입니다.
 - `connect_mlflow.py`: mlflow를 통해 모델 학습을 모니터링하고, 최적의 성능을 보이는 모델의 weights를 사용하기 위해 mlflow 서버와 연결하는 스크립트입니다.


#### wcgan-gp
 - Wasserstein GAN: [paper](https://arxiv.org/pdf/1701.07875) | [github](https://github.com/martinarjovsky/WassersteinGAN)
 - Wasserstein GAN with Gradient Penalty: [paper](https://arxiv.org/abs/1704.00028) | [github](https://github.com/igul222/improved_wgan_training)
 - Conditional GAN: [paper](https://arxiv.org/abs/1411.1784) | [github](https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py)
 - 본 프로젝트의 목소리를 통한 얼굴 생성 결과 이미지의 사용자 평가를 위해 구현된 비교군(목소리x, 나이/성별o) 모델입니다.
 - Wasserstein GAN의 학습 안정성을 높이기 위해 gradient penalty를 추가하였고, 모델 결과를 유도하기 위한 condition을 추가하였습니다.
 - 모델 자체 성능이 비교 불가능한 수준으로 학습되어, celebA dataset을 통해 사전학습을 진행하고 이후 vox celeb dataset에 finetuning을 진행하였습니다.


  

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
