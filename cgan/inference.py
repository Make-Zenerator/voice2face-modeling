import torch
from model import ConditionalGenrator
import os
from PIL import Image
from datetime import datetime

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def gender_processing(input_gender: int, device='cpu') -> torch.Tensor:
    gender = 0
    if input_gender in ['m', 'male', 'man']:
        gender = 0
    elif input_gender in ['f', 'female', 'woman']:
        gender = 1
    gender_tensor = torch.zeros((1,1)).to(device)
    gender_tensor[0] = torch.tensor(gender)
    return gender_tensor

def age_processing(input_age: int):
    age_boundary = [3, 7, 14, 23, 36, 46, 58, 121]
    for i in range(len(age_boundary)):
        if input_age < age_boundary[i]:
            age = i
            break
    
    if age >= 8:
        age = 7
    
    return age

def condition_processing(gender: torch.Tensor, age: int, device='cpu') -> torch.Tensor:
    age_condition = torch.zeros((1,8)).to(device)
    age_condition[0][age] = 1
    condition = torch.cat([gender, age_condition], dim=1).to(device)
    return condition

def save_image(image, path):

    if not os.path.exists(path):
        os.mkdir(path)

    img = Image.open(image)
    img.save(os.path.join(path, f'{start_time}.jpg'), "JPEG")

def main(input_gender, input_age):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "output/1709112384.74361/generator_110.pth"
    output_path = "results/condition_result"
    noise_dim = 100
    condition_dim = 9

    model = ConditionalGenrator(noise_dim=noise_dim, condition_dim=condition_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    print(model.parameters)
    model.eval()

    noise = torch.randn(1, noise_dim, device=device)
    gender = gender_processing(input_gender, device=device)
    age = age_processing(input_age)
    condition = condition_processing(gender, age, device=device)

    result = model(noise, condition)
    save_image(result, output_path)
    return result

if __name__ == "__main__":
    _ = main('f', 25)
    
    