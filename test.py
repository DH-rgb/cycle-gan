import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Generator,Discriminator,init_weights
from utils import ImagePool,BasicDataset

import argparse
import time 
import os
import sys
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from memory_profiler import profile
import pdb 


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation: CycleGAN')
    #for train
    parser.add_argument('--image_size', '-i', type=int, default=256, help='input image size')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--epoch_decay', '-ed', type=int, default=100,
                        help='Number of epochs to start decaying learning rate to zero')                    
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--pool_size', type=int, default=50, help='for discriminator: the size of image buffer that stores previously generated images')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Assumptive weight of cycle consistency loss')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    #for save and load
    parser.add_argument('--sample_frequecy', '-sf', type=int, default=1000,
                        help='Frequency of taking a sample')
    parser.add_argument('--checkpoint_frequecy', '-cf', type=int, default=1,
                        help='Frequency of taking a checkpoint')
    parser.add_argument('--data_name', '-d', default="horse2zebra", help='Dataset name')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--log_dir', '-l', default='logs/',
                        help='Directory to output the log')
    parser.add_argument('--model', '-m', help='Model name')
    args = parser.parse_args()



    #set GPU or CPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    #set depth of resnet
    if args.image_size == 128:
        res_block=6
    else:
        res_block=9
    
    #set models
    G_A2B = Generator(3,res_block).to(device)
    G_B2A = Generator(3,res_block).to(device)
    D_A = Discriminator(3).to(device)
    D_B = Discriminator(3).to(device)

    # data pararell
    # if device == 'cuda':
    #     G_A2B = torch.nn.DataParallel(G_A2B)
    #     G_B2A = torch.nn.DataParallel(G_B2A)
    #     D_A = torch.nn.DataParallel(D_A)
    #     D_B = torch.nn.DataParallel(D_B)
    #     torch.backends.cudnn.benchmark=True

    #load parameters
    G_A2B.load_state_dict(torch.load("models/"+args.model+"/G_A2B/"+str(args.epoch-1)+".pth"))
    G_B2A.load_state_dict(torch.load("models/"+args.model+"/G_B2A/"+str(args.epoch-1)+".pth"))
    D_A.load_state_dict(torch.load("models/"+args.model+"/D_A/"+str(args.epoch-1)+".pth"))
    D_B.load_state_dict(torch.load("models/"+args.model+"/D_B/"+str(args.epoch-1)+".pth"))

    test_dataset = BasicDataset(args.data_name, args.image_size, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    with torch.no_grad():
        if not os.path.exists("result/" + args.model):
            os.makedirs("result/" + args.model)
        for i, data in enumerate(test_loader):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            trans_B, trans_A = G_B2A(real_B), G_A2B(real_A)
            rec_A, rec_B = G_B2A(trans_A), G_A2B(trans_B)

            image_A = torch.cat((real_A, trans_A, rec_A),0)
            image_B = torch.cat((real_B, trans_B, rec_B),0)
            save_image(image_A,"result/" + args.model + "/A_" + str(i) + ".png", nrow=3, normalize=True)
            save_image(image_B,"result/" + args.model + "/B_" + str(i) + ".png", nrow=3, normalize=True)
            
            sys.stdout.write(f"\r[Number {i+1}/{len(test_loader)}]")

if __name__ == "__main__":
    main()