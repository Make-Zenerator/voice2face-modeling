import logging
from logging import handlers
import os
import numpy as np

def pretty(d, indent=0):
    for key, value in d.items():
        print(f"{key}: {value}")

class MetricsLogger:
    def __init__(self, wandb_config):
        self.wandb_config = wandb_config

        self.metrics_scores = [[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
        self.metrics_counter = [[[[0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]

        self.gender_map = {'m': 0, 'f': 1}
        self.metrics_map = {0: 'psnr', 1: 'ssim', 2: 'mssim', 3: 'lpips', 4: 'fid'}
        self.noise_map = {0: 'clean', 1: 'noise1', 2: 'noise2', 3: 'noise3', 4: 'noise4', 5: 'noise5'}

    def set_idx(self, labels):
        self.gender_idx = self.gender_map[labels['gender']]
        self.age_idx = labels['age'] - 1
        self.noise_idx = labels['noise'] - 1

    def compute_metrics_scores(self):
        avg_scores = [[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]
        for i in range(2):
            for j in range(6):
                for k in range(6):
                    for v in range(5):
                        if self.metrics_counter[i][j][k][v] != 0:
                            avg_scores[i][j][k][v] = self.metrics_scores[i][j][k][v] / self.metrics_counter[i][j][k][v]
        return avg_scores

    def append_metrics(self, batch_size, labels, metrics_scores):
        for i in range(batch_size):
            self.set_idx(labels[i])
            # print(f"gender idx : {self.gender_idx}")
            # print(f"age idx : {self.age_idx}")
            # print(f"noise idx : {self.noise_idx}")
            # print(f"mterics scores : {metrics_scores[i]}")
            for idx, score in enumerate(metrics_scores[i]):
                # print(idx)
                # print(score)
                # print(np.shape(self.metrics_scores))
                # print(self.metrics_scores[self.gender_idx][self.age_idx][self.noise_idx][idx])
                self.metrics_scores[self.gender_idx][self.age_idx][self.noise_idx][idx] += score
                self.metrics_counter[self.gender_idx][self.age_idx][self.noise_idx][idx] += 1

    def init_metrics_scores(self):
        self.metrics_scores = [[[[[0.0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]]
        self.metrics_counter = [[[[[0 for _ in range(5)] for _ in range(6)] for _ in range(6)] for _ in range(2)]]

    def wandb_logging(self, epoch):
        avg_scores = self.compute_metrics_scores()
        wandb_log = {}
        for i in range(2):
            for j in range(6):
                for k in range(6):
                    for v in range(5):
                        if self.metrics_counter[i][j][k][v] != 0:
                            name = f"{list(self.gender_map.keys())[list(self.gender_map.values()).index(i)]}_{(j+1)*10}_{self.noise_map[k]}_{self.metrics_map[v]}"
                            value = avg_scores[i][j][k][v]
                            wandb_log[name] = value
        pretty(wandb_log)
        # need to complete wandb logging console
        # wandb.log(wandb_log)
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
