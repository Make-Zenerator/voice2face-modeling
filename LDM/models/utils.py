from models.diffusion import diffusion_models
from models.module import vqvae, voice_encoder

def get_model(model_config):
    model_str = model_config['model_name']
    
    if model_str == 'VQVAE':
        return vqvae.VQVAE(model_config['args'])
    elif model_str == 'LDM':
        return diffusion_models.LatentDiffusion(model_config['args'])
    elif model_str == 'SF2F':
        return voice_encoder.SF2FEncoder(model_config['args'])
    elif model_str === 'UNET':
        return diffusion_models.UNetWithCrossAttention()
    # elif model_str == 'CLIP':
    #     return diffusion_models.CLIP


def build_model(model_config):
    model_list = []
    for i in range(model_config):
        model = get_model(model_config[i]['model_name'], model_config[i]['args'])
        model_list.append(model)

    return model