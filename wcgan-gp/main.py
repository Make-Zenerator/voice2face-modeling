import argparse
import torch.backends.cudnn as cudnn

import os
from train import Train
from inference import Inference
from utils import Parser

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the WGAN-GP network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')
parser.add_argument('--use_gpu', default=True, dest="use_gpu")
parser.add_argument('--mode', default='inference', choices=['train', 'finetune', 'inference'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')
parser.add_argument('--checkpoint_path', default='./checkpoints', dest='checkpoint_path')
parser.add_argument('--checkpoint_name', default="", dest="checkpoint_name")
parser.add_argument('--checkpoint_fine_tune_name', default="fine_tune", dest="checkpoint_fine_tune_name")
parser.add_argument('--dataset_path', type=str, default='/workspace/data', dest='dataset_path')
parser.add_argument('--dataset_name', default='celeba', dest='dataset_name')
parser.add_argument('--dataset_fine_tune_name', default='Voxceleb', dest='dataset_fine_tune_name')
parser.add_argument('--output_path', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=100, dest='num_epoch')
parser.add_argument('--num_epochs_decay', type=int, default=100, dest='n_epochs_decay')
parser.add_argument('--num_freq_save', type=int,  default=5, dest='num_freq_save')
parser.add_argument('--batch_size', type=int, default=128, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=2e-4, dest='lr_G')
parser.add_argument('--lr_D', type=float, default=2e-4, dest='lr_D')
parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'], dest='lr_policy')
parser.add_argument('--lr_G_weight_decay', type=float, default=1e-5, dest='lr_G_weight_decay')
parser.add_argument('--lr_D_weight_decay', type=float, default=1e-5, dest='lr_D_weight_decay')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--fine_tune', type=bool, default=False, dest='fine_tune')
parser.add_argument('--fine_tune_num_epoch', type=int, default=100, dest='fine_tune_num_epoch')
parser.add_argument('--fine_tune_num_freq_save', type=int, default=1, dest='fine_tune_num_freq_save')

# for inference
parser.add_argument('--input_gender', default='m', choices=['m', 'man', 'male', 'f', 'female', 'woman'], dest='input_gender')
parser.add_argument('--input_age', type=int, default=25, dest='input_age')

PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train()
    elif ARGS.mode == 'finetune':
        TRAINER.fine_tuning()
    elif ARGS.mode == 'inference':
        output_path = Inference.inference()
        return output_path

if __name__ == '__main__':
    main()