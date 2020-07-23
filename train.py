import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from model import Generator,Discriminator,init_weights
from utils import ImagePool,UnalignedDataset

import argparse
import time 
import os
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import pdb 


class loss_scheduler():
    def __init__(self, args):
        self.epoch_decay = args.epoch_decay

    def f(self, epoch):
        #ベースの学習率に対する倍率を返す(pytorch仕様)
        if epoch<=self.epoch_decay:
            return 1
        else:
            scaling = 1 - (epoch-self.epoch_decay)/float(self.epoch_decay)
            return scaling


def imshow(img):
    npimg = img.numpy()
    npimg = 0.5 * (npimg + 1)  # [-1,1] => [0, 1]
    # [c, h, w] => [h, w, c]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def debug_test(train_loader):
    batch = iter(train_loader).next()
    print(batch['A'].shape)
    print(batch['B'].shape)
    print(batch['path_A'])
    print(batch['path_B'])
    images_A = batch['A']  # horses
    images_B = batch['B']  # zebras

    plt.figure(figsize=(10, 20))

    plt.subplot(1, 2, 1)
    imshow(make_grid(images_A, nrow=4))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    imshow(make_grid(images_B, nrow=4))
    plt.axis('off')



def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation: CycleGAN')
    #for train
    parser.add_argument('--image_size', '-i', type=int, default=256, help='input image size')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--epoch_decay', '-ed', type=int, default=100,
                        help='Number of epochs to start decaying learning rate to zero')                    
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--pool_size', type=int, default=50, help='for discriminator: the size of image buffer that stores previously generated images')
    parser.add_argument('--lambda_cycle', type=float, default=10, help='Assumptive weight of cycle consistency loss')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    #for save and load
    parser.add_argument('--sample_frequecy', '-sf', type=int, default=100,
                        help='Frequency of taking a sample')
    parser.add_argument('--checkpoint_frequecy', '-cf', type=int, default=1,
                        help='Frequency of taking a checkpoint')
    parser.add_argument('--dataset', '-d', help='Dataset name')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', help='Model name')
    args = parser.parse_args()



    #set GPU or CPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'

    #set depth of resnet
    if args.image_size == 128:
        res_block=6
    else:
        res_block=9
    
    #set models
    G_A2B = Generator(args.image_size,res_block).to(device)
    G_B2A = Generator(args.image_size,res_block).to(device)
    D_A = Discriminator(args.image_size).to(device)
    D_B = Discriminator(args.image_size).to(device)

    #init weights
    G_A2B.apply(init_weights)
    G_B2A.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    #set loss functions
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()

    #set optimizers
    optimizer_G = torch.optim.Adam(chain(G_A2B.parameters(),G_B2A.parameters()),lr=args.lr,betas=(args.beta1,0.999))
    optimizer_D = torch.optim.Adam(chain(D_A.parameters(),D_B.parameters()), lr=args.lr,betas=(args.beta1,0.999))
    
    scheduler_G = LambdaLR(optimizer_G,lr_lambda=loss_scheduler(args).f)
    scheduler_D = LambdaLR(optimizer_D,lr_lambda=loss_scheduler(args).f)

    #dataset loading
    train_dataset = UnalignedDataset(args.image_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #######################################################################################
    debug_test(train_loader)
    pdb.set_trace()

    #train
    total_epoch = args.epoch

    fake_A_buffer = ImagePool()
    fake_B_buffer = ImagePool()

    for epoch in range(total_epoch):
        for i, data in enumerate(dataset):
            #generate image
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            fake_A, fake_B = G_B2A(real_B), G_A2B(real_A)
            rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)

            #train generator
            optimizer_G.zero_grad()

            pred_fake_A = D_A(fake_A)
            loss_G_B2A = adv_loss(pred_fake_A, torch.tensor(1).expand_as(pred_fake_A))
            
            pred_fake_B = D_B(fake_B)
            loss_G_A2B = adv_loss(pred_fake_B, torch.tensor(1).expand_as(pred_fake_B))

            loss_cycle_A = cycle_loss(rec_A, real_A)
            loss_cycle_B = cycle_loss(rec_B, real_B)

            loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*lambda_cycle + loss_cycle_B*lambda_cycle
            loss_G.backward()
            optimizer_G.step()

            #train discriminator
            optimizer_D.zero_grad()

            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A_buffer.get_images(fake_A).detach())
            loss_D_A_real = adv_loss(pred_real_A, torch.tensor(1).expand_as(pred_real_A))
            loss_D_A_fake = adv_loss(pred_fake_A, torch.tensor(0).expand_as(pred_fake_A))
            loss_D_A = (loss_D_A_fake + loss_D_A_real)*0.5
            loss_D_A.backward()

            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B_buffer.get_images(fake_B).detach())
            loss_D_B_real = adv_loss(pred_real_B, torch.tensor(1).expand_as(pred_real_B))
            loss_D_B_fake = adv_loss(pred_fake_B, torch.tensor(0).expand_as(pred_fake_B))
            loss_D_B = (loss_D_B_fake + loss_D_B_real)*0.5
            loss_D_B.backward()

            optimizer_D.step()

            #get sample
            if (epoch * len(train_dataloader) + i)&sample_frequency ==0:
                # real_A_sample = real_A.cpu().detach().numpy()[0]
                # real_B_sampel = real_B.cpu().detach().numpy()[0]
                # fake_A_sample = fake_A.cpu().detach().numpy()[0]
                # fake_B_sample = fake_B.cpu().detach().numpy()[0]
                # rec_A_sample = rec_A.cpu().detach().numpy()[0]
                # rec_B_sample = rec_B.cpu().detach().numpy()[0]
                images_sample = torch.cat((real_A.data, fake_A.data, rec_A.data, real_B.data, fake_B.data, rec_B.data),0)
                save_image(images_sample, "sample/" + model + "/" + str(epoch * len(train_dataloader) + i) + ".png", nrow=5, normalize=True)
    
            
        #update learning rate
        scheduler_G.step()
        scheduler_D.step()
        
        if epoch % opts.checkpoint_every == 0:
            torch.save(G_A2B.state_dict(), "models/"+model+"/G_A2B/"+str(epoch)+".pth")
            torch.save(G_B2A.state_dict(), "models/"+model+"/G_B2A/"+str(epoch)+".pth")
            torch.save(D_A.state_dict(), "models/"+model+"/D_A/"+str(epoch)+".pth")
            torch.save(D_B.state_dict(), "models/"+model+"/D_B/"+str(epoch)+".pth")




if __name__ == "__main__":
    main()