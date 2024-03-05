import os
import torch
from model import ConditionalGenrator


def inference(self):
    mode = self.mode

    batch_size = self.batch_size
    device = self.device
    gpu_ids = self.gpu_ids

    ny_in = self.ny_in
    nx_in = self.nx_in

    nch_in = self.nch_in
    nch_out = self.nch_out
    nch_ker = self.nch_ker

    norm = self.norm

    name_data = self.name_data
    input_gender = self.input_gender
    input_age = self.input_age
    ## setup dataset
    dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

    dir_result = os.path.join(self.dir_result, self.scope, name_data)
    dir_result_save = os.path.join(dir_result, 'images')
    if not os.path.exists(dir_result_save):
        os.makedirs(dir_result_save)

    condition_dim = 9
    ## setup network
    netG = ConditionalGenrator(nch_in=nch_in, condition_dim=9, nch_ker=64)

    ## load from checkpoints
    st_epoch = 0

    ckpt = os.listdir(dir_chck)
    ckpt.sort()

    for i in range(len(ckpt)):
        ## test phase
        model_path = os.path.join(dir_chck, ckpt[i])
        netG = self.load(model_path, netG, mode=mode)
        with torch.no_grad():

            netG.eval()

            input = torch.randn(1, nch_in).to(device)
            noise = torch.randn(1, nch_in, device=device)
            gender = gender_processing(input_gender, device=device)
            age = age_processing(input_age)
            condition = condition_processing(gender, age, device=device)
            # input = torch.cat([input, condition], dim=1)
            # input = input.unsqueeze(2).unsqueeze(2)

            output = netG(input, condition)
            min_ = output.min()
            max_ = output.max()
            clipping = ((output-min_)/(max_-min_))

            output_path = "results/cgan/images"

            self.save_image(clipping, output_path, i)


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