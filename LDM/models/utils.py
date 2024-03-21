from models.diffusion import diffusion_models
from models.module import vqvae, voice_encoder

def get_model(model_str: str):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """
    if model_str == 'VQVAE':
        return vqvae.VQVAE
    elif model_str == 'LDM':
        return diffusion_models.LatentDiffusion
    elif model_str == 'SF2F':
        return voice_encoder.SF2FEncoder
    # elif model_str == 'CLIP':
    #     return diffusion_models.CLIP