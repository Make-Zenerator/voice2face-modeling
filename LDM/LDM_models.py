import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from diffusers import (
    UNet2DConditionModel,
    DDIMScheduler,
    AutoencoderKL
)
from custom_pipeline import LDMSpeechToFacePipeline
from sf2f_encoder import SF2FEncoder
from peft import LoraConfig, get_peft_model


repo_id = 'ComVis/ldm-text2im-large-256'

def build_pretrained_models(repo_id):

    unet = UNet2DConditionModel.from_pretrained(repo_id, shubfolder='unet', safetensors=True)
    vqvae = AutoencoderKL.from_pretrained(repo_id, subfolder='vqvae')
    scheduler = DDIMScheduler.from_pretrained(repo_id, subfolder='scheduler')
    return {'unet': unet, 'vqvae': vqvae, 'scheduler': scheduler}

def build_voice_encoder(input_channel=40, output_channels=512):
    sf2f_encoderr_kwargs = {
        'input_channel': input_channel,
        'channels': [256, 384, 576, 864],
        'output_channels': output_channels,
        'add_noise': False,
        'normalize': True,
        'return_seq': False,
        'inception_mode': True,
    }

    voice_encoder = SF2FEncoder(sf2f_encoderr_kwargs)
    return voice_encoder

def build_speech_to_face_pipeline(config):
    pretrained_models = build_pretrained_models(config["repo_id"])
    voice_encoder = build_voice_encoder(input_channel=config["input_channel"], output_channels=config["output_channels"])
    s2f_pipeline = LDMSpeechToFacePipeline(
                                        unet=pretrained_models["unet"],
                                        vqvae=pretrained_models["vqvae"],
                                        sf2f=voice_encoder,
                                        scheduler=pretrained_models["scheduler"],
                                        )
    return s2f_pipeline

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}" 
    )

def build_Lora_model(config, target_model):
    lora_config = LoraConfig(
        r=config['r'],
        lora_alpha=config['lora_alpha'],
        target_modeuls=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias=config['bias'],
        modules_to_save=config['modules_to_save'],
    )
    lora_model = get_peft_model(target_model, lora_config)
    print_trainable_parameters(lora_model)
    return lora_model