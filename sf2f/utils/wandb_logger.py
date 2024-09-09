import logging
from logging import handlers
import os
import numpy as np
import wandb
import mlflow
import torch

def pretty(d, indent=0):
    for key, value in d.items():
        print(f"{key}: {value}")

class MetricsLogger:
    def __init__(self, wandb_config):
        self.wandb_config = wandb_config
        self.samples = []

        if self.wandb_config == 'olk':
            self.metrics_scores = [[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
            self.metrics_counter = [[[[0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
        else:
            self.gender_scores = [[0.0 for _ in range(5)] for _ in range(2)]
            self.gender_counter = [[0 for _ in range(5)] for _ in range(2)]

        self.psnr_score = 0.0
        self.ssim_score = 0.0
        self.mssim_score = 0.0
        self.lpips_score = 0.0
        self.fid_score = 0.0

        self.len_of_data = 0


        self.gender_map = {'m': 0, 'M': 0, 'f': 1, 'F': 1}
        self.metrics_map = {0: 'psnr', 1: 'ssim', 2: 'mssim', 3: 'lpips', 4: 'fid'}
        self.noise_map = {0: 'clean', 1: 'noise1', 2: 'noise2', 3: 'noise3', 4: 'noise4', 5: 'noise5'}

    def set_idx(self, labels):
        if self.wandb_config == 'olk':
            self.gender_idx = [self.gender_map[gender] for gender in labels['gender']]
            self.age_idx = labels['age'] - 1
            self.noise_idx = labels['noise'] - 1
        else:
            self.gender_idx = self.gender_map[labels]

    def compute_metrics_scores(self):
        if self.wandb_config == 'olk':
            avg_scores = [[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
            for i in range(2):
                for j in range(6):
                    for k in range(6):
                        for v in range(5):
                            if self.metrics_counter[i][j][k][v] != 0:
                                avg_scores[i][j][k][v] = self.metrics_scores[i][j][k][v] / self.metrics_counter[i][j][k][v]
        else:
            avg_scores = [[0.0 for _ in range(5)] for _ in range(2)]
            for i in range(2):
                for j in range(5):
                    if self.gender_counter[i][j] != 0:
                        avg_scores[i][j] = self.gender_scores[i][j] / self.gender_counter[i][j]

        return avg_scores

    def append_metrics(self, batch_size, labels, metrics_scores):
        if self.wandb_config == 'olk':
            self.set_idx(labels)
            for i in range(batch_size):
                self.metrics_scores[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][0] += metrics_scores[0][0][i]
                self.metrics_scores[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][1] += metrics_scores[1][0][i]
                self.metrics_scores[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][2] += metrics_scores[2][0][i]
                self.metrics_scores[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][3] += metrics_scores[3][0][i]
                self.metrics_scores[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][4] += metrics_scores[4][0][i]
                self.metrics_counter[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][0] += 1
                self.metrics_counter[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][1] += 1
                self.metrics_counter[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][2] += 1
                self.metrics_counter[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][3] += 1
                self.metrics_counter[self.gender_idx[i]][self.age_idx[i]][self.noise_idx[i]][4] += 1

                self.psnr_score += metrics_scores[0][0][i]
                self.ssim_score += metrics_scores[1][0][i]
                self.mssim_score += metrics_scores[2][0][i]
                self.lpips_score += metrics_scores[3][0][i]
                self.fid_score += metrics_scores[4][0][i]

        elif self.wandb_config == 'vox':
            for i in range(batch_size):
                self.set_idx(labels[i])
                self.gender_scores[self.gender_idx][0] += metrics_scores[0][0][i]
                self.gender_scores[self.gender_idx][1] += metrics_scores[1][0][i]
                self.gender_scores[self.gender_idx][2] += metrics_scores[2][0][i]
                self.gender_scores[self.gender_idx][3] += metrics_scores[3][0][i]
                self.gender_scores[self.gender_idx][4] += metrics_scores[4][0][i]
                self.gender_counter[self.gender_idx][0] += 1
                self.gender_counter[self.gender_idx][1] += 1
                self.gender_counter[self.gender_idx][2] += 1
                self.gender_counter[self.gender_idx][3] += 1
                self.gender_counter[self.gender_idx][4] += 1


                self.psnr_score += metrics_scores[0][0][i]
                self.ssim_score += metrics_scores[1][0][i]
                self.mssim_score += metrics_scores[2][0][i]
                self.lpips_score += metrics_scores[3][0][i]
                self.fid_score += metrics_scores[4][0][i]

        self.len_of_data += batch_size


    def init_metrics_scores(self):
        if self.wandb_config == 'olk':
            self.metrics_scores = [[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
            self.metrics_counter = [[[[0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
        else:
            self.gender_scores = [[0.0 for _ in range(5)] for _ in range(2)]
            self.gender_counter = [[0 for _ in range(5)] for _ in range(2)]
        torch.cuda.empty_cache()
        self.imgs = []
        self.samples = []

    def append_samples(self, epoch, samples, labels):
        self.samples.append(wandb.Image(samples, caption=f"Pred:{epoch}_{labels}"))

    def wandb_logging(self, epoch, sub_dataset):
        avg_scores = self.compute_metrics_scores()
        wandb_log = {}
        wandb_log[f'{sub_dataset}_psnr'] = self.psnr_score / self.len_of_data
        wandb_log[f'{sub_dataset}_ssim'] = self.ssim_score / self.len_of_data
        wandb_log[f'{sub_dataset}_mssim'] = self.mssim_score / self.len_of_data
        wandb_log[f'{sub_dataset}_lpips'] = self.lpips_score / self.len_of_data
        wandb_log[f'{sub_dataset}_fid'] = self.fid_score / self.len_of_data

        if self.wandb_config == 'olk':
            for i in range(2):
                for j in range(6):
                    for k in range(6):
                        for v in range(5):
                            if self.metrics_counter[i][j][k][v] != 0:
                                name = f"{sub_dataset}_{list(self.gender_map.keys())[list(self.gender_map.values()).index(i)]}_{(j+1)*10}_{self.noise_map[k]}_{self.metrics_map[v]}"
                                value = avg_scores[i][j][k][v]
                                wandb_log[name] = value
        else:
            for i in range(2):
                for j in range(5):
                    if self.gender_counter[i][j] != 0:
                        name = f"{sub_dataset}_{list(self.gender_map.keys())[list(self.gender_map.values()).index(i)]}_{self.metrics_map[j]}"
                        value = avg_scores[i][j]
                        wandb_log[name] = value

        # mlflow.log_metrics(wandb_log)

        if len(self.samples) != 0:
            wandb_log['Pred_Images'] = self.samples

        # pretty(wandb_log)
        # need to complete wandb logging console
        wandb.log(
            wandb_log,
        )
        self.init_metrics_scores()

if __name__ == '__main__':
    import random
    wandb_config = None
    new_logger = MetricsLogger(wandb_config)
    random_age, random_noise = [], []   
    for i in range(10):
        random_age.append(random.randint(1, 6))
        random_noise.append(random.randint(1, 6))
    data = [{'gender': 'f', 'age': i, 'noise': j} for i, j in zip(random_age, random_noise)]

    # print(data)

    for row in data:
        print(row)

    rows, cols = 10, 5
    metrics_scores = [[random.uniform(0, 50) for _ in range(cols)] for _ in range(rows)]

    for row in metrics_scores:
        print(row)

    new_logger.append_metrics(10, data, metrics_scores)
    new_logger.wandb_logging(epoch=1)
