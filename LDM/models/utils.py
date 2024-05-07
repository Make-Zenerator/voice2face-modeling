from models.diffusion.diffusion_models import UNetWithCrossAttention, LatentDiffusion, get_betas
from models.module import vqvae, voice_encoder

def get_model(model_config):
    model_str = model_config['model_name']
    
    for i in range(len(model_str)):
            
        if model_str == 'VQVAE':
            autoencoder = vqvae.VQVAE(model_config[model_str]['args'])
        elif model_str == 'SF2F':
            voice_model = voice_encoder.SF2FEncoder(model_config[model_str]['args'])
        elif model_str == 'UNET':
            unet = UNetWithCrossAttention(model_config[model_str]['args'])
    # elif model_str == 'CLIP':
    #     return diffusion_models.CLIP
    ldm = LatentDiffusion(unet=unet, 
                          voice_encoder=voice_model, 
                          vqvae=autoencoder, 
                          betas=get_betas(model_config['timestepts'],
                                          scheduler_name=model_config['scehdule_name'])
                         )
    return ldm

# def build_model(model_config):
#     config_args = model_config
#     Unet = get_model(config_args[0])
#     voice_encoder = get_model(config_args[1])
#     vqvae = get_model(config_args[2])
#     model = diffusion_models.LatentDiffusion(unet=Unet, voice_encoder=voice_encoder, vqvae=vqvae)

#     return model