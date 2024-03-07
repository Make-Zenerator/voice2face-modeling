from models import (DCGAN,
                    Discriminator,
                    init_net,
                    ConditionMLPGAN,
                    ConditionEmbeddingGAN,
                    ConditioEmbeddingDiscriminator,
                    UNet,
                    ConditionalUNetGAN,
                    ConditionConvGenerator,
                    ConditionFinetuneGAN,
                    ConditionLinearFinetuneGAN,
                    ConditionLinearFinetuneDiscriminator
                    )
from dataset import HQVoxceleb, CelebADataset
import torchvision.datasets as datasets
from layer import GradientPaneltyLoss

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.transforms.functional import to_pil_image

from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statistics import mean

from datetime import datetime

class Train:
    def __init__(self, args):
        self.mode = args.mode
        self.train_continue = args.train_continue

        self.scope = args.scope
        self.dir_checkpoint = args.dir_checkpoint
        self.dir_log = args.dir_log

        self.dir_data = args.dir_data
        self.dir_result = args.dir_result

        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.wgt_gan = args.wgt_gan
        self.wgt_disc = args.wgt_disc

        self.optim = args.optim
        self.beta1 = args.beta1

        self.ny_in = args.ny_in
        self.nx_in = args.nx_in
        self.nch_in = args.nch_in

        self.ny_load = args.ny_load
        self.nx_load = args.nx_load
        self.nch_load = args.nch_load

        self.ny_out = args.ny_out
        self.nx_out = args.nx_out
        self.nch_out = args.nch_out

        self.nch_ker = args.nch_ker

        self.data_type = args.data_type
        self.norm = args.norm

        self.gpu_ids = args.gpu_ids

        self.num_freq_disp = args.num_freq_disp
        self.num_freq_save = args.num_freq_save

        self.name_data = args.name_data

        self.input_gender = args.input_gender
        self.input_age = args.input_age
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.fine_tune = args.fine_tune
        self.fine_tune_num_epoch = args.fine_tune_num_epoch
        self.fine_tune_num_freq_save = args.fine_tune_num_freq_save

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % self.gpu_ids[0])
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, dir_chck, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(os.path.join(dir_chck,self.start_time)):
            os.makedirs(os.path.join(dir_chck,self.start_time))

        torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                   '%s/%s/model_epoch%04d.pth' % (dir_chck, self.start_time, epoch))

    def load(self, ckpt, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):

        dict_net = torch.load(ckpt)

        print(f'Loaded {ckpt}')

        if mode == 'train':
            netG.load_state_dict(dict_net['netG'])
            netD.load_state_dict(dict_net['netD'])
            optimG.load_state_dict(dict_net['optimG'])
            optimD.load_state_dict(dict_net['optimD'])
            st_epoch = ckpt.split("epoch")[1]
            st_epoch = int(st_epoch.split(".pth")[0])
            return netG, netD, optimG, optimD, st_epoch

        elif mode == 'test' or mode == 'inference':
            netG.load_state_dict(dict_net['netG'])

            return netG
        
    def custom_collate_fn(self, batch):
        images, gender_labels, age_labels = zip(*batch)
        images = torch.stack(images, dim=0).float()  # 이미지를 torch.Tensor로 변환하고 float로 변환
        gender_labels = torch.tensor(gender_labels)
        age_labels = torch.tensor(age_labels)
        return images, gender_labels, age_labels
    
    def train(self):
        mode = self.mode

        train_continue = self.train_continue
        num_epoch = self.num_epoch

        lr_G = self.lr_G
        lr_U = self.lr_G
        lr_D = self.lr_D

        wgt_gan = self.wgt_gan
        wgt_disc = self.wgt_disc

        batch_size = self.batch_size
        device = self.device

        gpu_ids = self.gpu_ids

        nch_in = self.nch_in
        nch_out = self.nch_out
        nch_ker = self.nch_ker
        condition_dim = 9

        norm = self.norm
        name_data = self.name_data

        num_freq_disp = self.num_freq_disp
        num_freq_save = self.num_freq_save

        ny_in = self.ny_in
        nx_in = self.nx_in

        ## setup dataset
        dir_chck = os.path.join(self.dir_checkpoint, self.scope, name_data)

        dir_data_train = os.path.join(self.dir_data, name_data)
        dir_log = os.path.join(self.dir_log, self.scope, name_data)

        fine_tune = self.fine_tune
        fine_tune_num_epoch = self.fine_tune_num_epoch
        fine_tune_num_freq_save = self.fine_tune_num_freq_save

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        # dataset = HQVoxceleb(transform=transform)
        dataset = CelebADataset(transform=transform)
        # train_dataset, val_dataset = dataset.split_dataset()

        # dataset = CelebADataset(transform=transform)
        # dataset = datasets.CelebA(root="/workspace/CelebA", transform=transform, download=True)

        # loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, collate_fn=self.custom_collate_fn)
        loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        num_train = len(dataset)

        num_batch_train = int((num_train / batch_size) + ((num_train % batch_size) != 0))

        ## setup network
        # netG = DCGAN(nch_in+condition_dim, nch_out, nch_ker, norm)
        # netG = ConditionMLPGAN(nch_in=(nch_in+condition_dim))
        # netG = ConditionConvGenerator()
        # netG = ConditionFinetuneGAN(nch_in=nch_in)
        # netG = ConditionEmbeddingGAN(nch_in=nch_in, condition_dim=9, nch_ker=64)
        # netG = ConditionalUNetGAN(noise_dim=100, nch_in=3, condition_dim=9, nch_out=3)
        netG = ConditionLinearFinetuneGAN(nch_in=nch_in)
        # netD = Discriminator(nch_out, nch_ker, [])
        netD = ConditionLinearFinetuneDiscriminator()
        # netD = ConditioEmbeddingDiscriminator(input_size=(self.nch_out,self.nx_out,self.ny_out), nch_ker=64, condition_dim=9)

        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## setup loss & optimization
        fn_GP = GradientPaneltyLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        # schedG = get_scheduler(optimG, self.opts)
        # schedD = get_scheduler(optimD, self.opts)

        # schedG = torch.optim.lr_scheduler.ExponentialLR(optimG, gamma=0.9)
        # schedD = torch.optim.lr_scheduler.ExponentialLR(optimD, gamma=0.9)

        ## load from checkpoints
        st_epoch = 0

        if train_continue == 'on':
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            file_path = os.path.join(dir_chck, ckpt[0])
            netG, netD, optimG, optimD, st_epoch = self.load(file_path, netG, netD, optimG, optimD, mode=mode)

        for epoch in range(st_epoch + 1, num_epoch + 1):
            ## training phase
            netG.train()
            netD.train()

            loss_G_train = []
            loss_D_real_train = []
            loss_D_fake_train = []

            for i, data in enumerate(loader_train):
                def should(freq):
                    return freq > 0 and (i % freq == 0 or i == num_batch_train)

                # images, gender_labels, age_labels = data
                images = data.to(device)
                batch_size = images.size(0)
                # age_labels = age_labels.unsqueeze(1)
                # age_condition = torch.zeros(batch_size, 8)
                # age_condition.scatter_(1, age_labels, 1)
                # gender_labels = gender_labels.unsqueeze(1)
                # condition = torch.cat([gender_labels, age_condition], dim=1).to(device) 
                input = torch.randn(batch_size, nch_in).to(device)
                # input = torch.randn(batch_size, nch_in).to(device)
                # input = torch.cat([input, condition], dim=1)
                # input = input.unsqueeze(2).unsqueeze(2) # (batch_size, noize+condition, 1, 1)
                # forward netG
                # output = netG(input, condition)
                output = netG(input)

                # backward netD
                set_requires_grad(netD, True)
                optimD.zero_grad()

                # pred_real = netD(images, condition)
                # pred_fake = netD(output.detach(), condition)
                pred_real = netD(images)
                pred_fake = netD(output.detach())

                alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
                output_ = (alpha * images + (1 - alpha) * output.detach()).requires_grad_(True)
                # src_out_ = netD(output_, condition)
                src_out_ = netD(output_)

                # BCE Loss
                # loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                # loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))

                # WGAN Loss
                loss_D_real = torch.mean(pred_real)
                loss_D_fake = -torch.mean(pred_fake)

                # Gradient penalty Loss
                loss_D_gp = fn_GP(src_out_, output_)

                loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_gp
                # loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward(retain_graph=True)
                optimD.step()


                set_requires_grad(netD, False)
                # backward netG
                optimG.zero_grad()
                
                # pred_fake = netD(output, condition)
                pred_fake = netD(output)

                loss_G = torch.mean(pred_fake)
                loss_G.backward()
                optimG.step()
                

                # get losses
                loss_G_train += [loss_G.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_fake_train += [loss_D_fake.item()]

                print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                      'GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f' %
                      (epoch, i, num_batch_train,
                       mean(loss_G_train), mean(loss_D_fake_train), mean(loss_D_real_train)))


            ## save
            if (epoch % num_freq_save) == 0:
                self.save(dir_chck, netG, netD, optimG, optimD, epoch)

        if fine_tune:
            
            dataset = HQVoxceleb(transform=transform)
            loader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, collate_fn=self.custom_collate_fn)

            netG = ConditionLinearFinetuneGAN(nch_in=nch_in)
            # netD = Discriminator(nch_out, nch_ker, [])
            netD = ConditionLinearFinetuneDiscriminator()
            init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
            init_net(netD, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

            ## setup loss & optimization
            fn_GP = GradientPaneltyLoss().to(device)

            paramsG = netG.parameters()
            paramsD = netD.parameters()

            optimG = torch.optim.Adam(paramsG, lr=1e-5, betas=(self.beta1, 0.999))
            optimD = torch.optim.Adam(paramsD, lr=1e-5, betas=(self.beta1, 0.999))

            # dir_chck = os.path.join(dir_chck, self.start_time)
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            file_path = os.path.join(dir_chck, ckpt[-3])
            netG, netD, optimG, optimD, st_epoch = self.load(file_path, netG, netD, optimG, optimD, mode=mode)

            new_sequence = nn.Sequential(
                nn.Linear(in_features=(nch_in+nch_ker//4+nch_ker//4), out_features=2*nch_ker),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )
            netG.fc1 = new_sequence
            new_sequence = nn.Sequential(
                nn.Conv2d(in_channels=5, out_channels=nch_ker, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nch_ker),
                nn.LeakyReLU(0.2)
            )
            netD.dsc1 = new_sequence
            
            netG.set_fine_tune(True)
            netD.set_fine_tune(True)
            netG = netG.to(device)
            netD = netD.to(device)

            dir_chck = os.path.join(dir_chck, "fine_tuning")
            for epoch in range(fine_tune_num_epoch + 1):
                ## training phase
                netG.train()
                netD.train()

                loss_G_train = []
                loss_D_real_train = []
                loss_D_fake_train = []

                for i, data in enumerate(loader_train):
                    def should(freq):
                        return freq > 0 and (i % freq == 0 or i == num_batch_train)

                    images, gender_labels, age_labels = data
                    images = images.to(device)
                    batch_size = images.size(0)
                    gender_labels = gender_labels.unsqueeze(1)
                    gender_conditions = torch.zeros(batch_size, 2)
                    gender_conditions = gender_conditions.scatter_(1, gender_labels, 1).to(device)
                    age_labels = age_labels.unsqueeze(1)
                    age_conditions = torch.zeros(batch_size, 8)
                    age_conditions = age_conditions.scatter_(1, age_labels, 1).to(device)      
                    # conditions = torch.cat([gender_conditions, age_conditions], dim=1).to(device) 
                    input = torch.randn(batch_size, nch_in).to(device)
                    # input = torch.cat([input, conditions], dim=1)
                    input_G = (input, gender_conditions, age_conditions)
                    input_D = (images, gender_conditions, age_conditions)
                    output = netG(input_G)
                    output_D = (output.detach(), gender_conditions, age_conditions)
                    
                    # backward netD
                    set_requires_grad(netD, True)
                    optimD.zero_grad()

                    pred_real = netD(input_D)
                    pred_fake = netD(output_D)

                    alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
                    output_ = (alpha * images + (1 - alpha) * output.detach()).requires_grad_(True)
                    src_out_ = netD((output_, gender_conditions, age_conditions))

                    # BCE Loss
                    # loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
                    # loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))

                    # WGAN Loss
                    loss_D_real = torch.mean(pred_real)
                    loss_D_fake = -torch.mean(pred_fake)

                    # Gradient penalty Loss
                    loss_D_gp = fn_GP(src_out_, output_)

                    loss_D = 0.5 * (loss_D_real + loss_D_fake) + loss_D_gp
                    # loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    loss_D.backward(retain_graph=True)
                    optimD.step()


                    set_requires_grad(netD, False)
                    # backward netG
                    optimG.zero_grad()
                    
                    # pred_fake = netD(output, condition)
                    pred_fake = netD((output, gender_conditions, age_conditions))

                    loss_G = torch.mean(pred_fake)
                    loss_G.backward()
                    optimG.step()
                    

                    # get losses
                    loss_G_train += [loss_G.item()]
                    loss_D_real_train += [loss_D_real.item()]
                    loss_D_fake_train += [loss_D_fake.item()]

                    print('TRAIN: EPOCH %d: BATCH %04d/%04d: '
                        'GEN GAN: %.4f DISC FAKE: %.4f DISC REAL: %.4f' %
                        (epoch, i, num_batch_train,
                        mean(loss_G_train), mean(loss_D_fake_train), mean(loss_D_real_train)))

                
                ## save
                if (epoch % fine_tune_num_freq_save) == 0:
                    self.save(dir_chck, netG, netD, optimG, optimD, epoch)

    def test(self):
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

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        condition_dim = 9
        ## setup network
        netG = DCGAN(nch_in+condition_dim, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)

        ## load from checkpoints
        st_epoch = 0

        netG, st_epoch = self.load(dir_chck, netG, mode=mode)

        ## test phase
        with torch.no_grad():
            netG.eval()
            # netG.train()

            # input = torch.randn(batch_size, nch_in).to(device)
            noise = torch.randn(1, nch_in, device=device)
            gender = gender_processing(input_gender, device=device)
            age = age_processing(input_age, device=device)
            # condition = condition_processing(gender, age, device=device)
            # input = torch.cat([input, condition], dim=1)
            # input = input.unsqueeze(2).unsqueeze(2)

            output = netG((noise, gender, age))

            output = transform(output)

            for j in range(output.shape[0]):
                name = j
                fileset = {'name': name,
                            'output': "%04d-output.png" % name}

                if nch_out == 3:
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze())
                elif nch_out == 1:
                    plt.imsave(os.path.join(dir_result_save, fileset['output']), output[j, :, :, :].squeeze(), cmap=cm.gray)

                append_index(dir_result, fileset)


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
        # netG = ConditionConvGenerator()
        # netG = ConditionLinearFinetuneGAN(nch_in=nch_in)
        netG = ConditionLinearFinetuneGAN(nch_in=nch_in)
        # netG = DCGAN(nch_in+condition_dim, nch_out, nch_ker, norm)
        init_net(netG, init_type='normal', init_gain=0.02, gpu_ids=gpu_ids)
        netG.set_fine_tune(self.fine_tune)
        ## load from checkpoints
        st_epoch = 0
        if self.fine_tune:
            new_sequence = nn.Sequential(
                nn.Linear(in_features=(nch_in+nch_ker//4+nch_ker//4), out_features=128),
                nn.ReLU(0.2),
                nn.Dropout(0.1)
            )
            netG.fc1 = new_sequence
            netG = netG.to(device)

        ckpt = os.listdir(dir_chck)
        ckpt.sort()

        # for i in range(len(ckpt)):
        #     ## test phase
        #     model_path = os.path.join(dir_chck, ckpt[i])
        #     netG = self.load(model_path, netG, mode=mode)
        #     with torch.no_grad():

        #         netG.eval()

        #         # input = torch.randn(1, nch_in).to(device)
        #         noise = torch.randn(1, nch_in, device=device)
        #         gender = gender_processing(input_gender, device=device)
        #         age = age_processing(input_age)
        #         age_conditions = torch.zeros(1, 8)
        #         age_conditions[0][age] = torch.Tensor(1)
        #         age_conditions = age_conditions.to(device)
        #         # age_conditions = age_conditions.scatter_(1, age, 1).to(device)
        #         # condition = condition_processing(gender, age, device=device)
        #         # input = torch.cat([input, condition], dim=1)
        #         # input = input.unsqueeze(2).unsqueeze(2)

        #         # output = netG(input, condition)
        #         output = netG((noise, gender, age_conditions))

        #         min_ = output.min()
        #         max_ = output.max()
        #         clipping = ((output-min_)/(max_-min_))

        #         output_path = "results/wgan-gp/condition_linear_finetune_gan_general_discriminator"

        #         self.save_image(clipping, output_path, i)

        import random
        data_list = random.sample(range(4, 15), 10)
        gender = ['m','f']
        gender_list = [random.choice(gender) for i in range(50)]
        age_list = [random.choice(data_list) for i in range(50)]

        model_path = os.path.join(dir_chck, ckpt[-1])
        netG = self.load(model_path, netG, mode=mode)
        with torch.no_grad():

            netG.eval()
            for i in range(len(gender_list)):
            ## test phase

                # input = torch.randn(1, nch_in).to(device)
                noise = torch.randn(1, nch_in, device=device)
                gender = gender_processing(gender_list[i], device=device)
                age = age_processing(age_list[i])
                age_conditions = torch.zeros(1, 8)
                age_conditions[0][age] = torch.Tensor(1)
                age_conditions = age_conditions.to(device)
                # age_conditions = age_conditions.scatter_(1, age, 1).to(device)
                # condition = condition_processing(gender, age, device=device)
                # input = torch.cat([input, condition], dim=1)
                # input = input.unsqueeze(2).unsqueeze(2)

                # output = netG(input, condition)
                output = netG((noise, gender, age_conditions))

                min_ = output.min()
                max_ = output.max()
                clipping = ((output-min_)/(max_-min_))

                output_path = "results/wgan-gp/condition_linear_finetune_gan_general_discriminator"

                self.save_image(clipping, output_path, i)
    
    
    def save_image(self, image, path, num):
        img = image.cpu().squeeze(0)
        img = to_pil_image(img)
        folder_path = os.path.join(path, self.start_time)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        img.save(os.path.join(folder_path, f'{num}.jpeg'), "JPEG")

def gender_processing(input_genders, device='cpu') -> torch.Tensor:
    gender_tensors = torch.zeros((1, 2))
    for i, input_gender in enumerate(input_genders):
        gender = 0
        if input_gender in ['m', 'male', 'man'] or input_gender == 0 :
            gender = 0
        elif input_gender in ['f', 'female', 'woman'] or input_gender == 1:
            gender = 1
        gender_tensors[i][gender] = torch.tensor(1)
    return gender_tensors.to(device)

def age_processing(input_age):
    if type(input_age) == str:
        input_age = int(input_age)
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

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def append_index(dir_result, fileset, step=False):
    index_path = os.path.join(dir_result, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        for key, value in fileset.items():
            index.write("<th>%s</th>" % key)
        index.write('</tr>')

    # for fileset in filesets:
    index.write("<tr>")

    if step:
        index.write("<td>%d</td>" % fileset["step"])
    index.write("<td>%s</td>" % fileset["name"])

    del fileset['name']

    for key, value in fileset.items():
        index.write("<td><img src='images/%s'></td>" % value)

    index.write("</tr>")
    return index_path


def add_plot(output, label, writer, epoch=[], ylabel='Density', xlabel='Radius', namescope=[]):
    fig, ax = plt.subplots()

    ax.plot(output.transpose(1, 0).detach().numpy(), '-')
    ax.plot(label.transpose(1, 0).detach().numpy(), '--')

    ax.set_xlim(0, 400)

    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    writer.add_figure(namescope, fig, epoch)
