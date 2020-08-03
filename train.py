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


def set_requires_grad(models, requires=False):
    if not isinstance(models,list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires



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
    parser.add_argument('--lambda_identity', type=float, default=0, help='Assumptive weight of identity mapping loss')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    #for save and load
    parser.add_argument('--sample_frequecy', '-sf', type=int, default=5000,
                        help='Frequency of taking a sample')
    parser.add_argument('--checkpoint_frequecy', '-cf', type=int, default=10,
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


    #init weights
    G_A2B.apply(init_weights)
    G_B2A.apply(init_weights)
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    #set loss functions
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    #set optimizers
    optimizer_G = torch.optim.Adam(chain(G_A2B.parameters(),G_B2A.parameters()),lr=args.lr,betas=(args.beta1,0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1,0.999))
    
    scheduler_G = LambdaLR(optimizer_G,lr_lambda=loss_scheduler(args).f)
    scheduler_D_A = LambdaLR(optimizer_D_A,lr_lambda=loss_scheduler(args).f)
    scheduler_D_B = LambdaLR(optimizer_D_B,lr_lambda=loss_scheduler(args).f)

    #dataset loading
    train_dataset = BasicDataset(args.data_name, args.image_size, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    #######################################################################################

    #train
    total_epoch = args.epoch

    fake_A_buffer = ImagePool()
    fake_B_buffer = ImagePool()

    for epoch in range(total_epoch):
        start = time.time()
        losses = [0 for i in range(6)]
        for i, data in enumerate(train_loader):
            #generate image
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            fake_A, fake_B = G_B2A(real_B), G_A2B(real_A)
            rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
            if args.lambda_identity>0:
                iden_A, iden_B = G_B2A(real_A), G_A2B(real_B)

            #train generator
            set_requires_grad([D_A,D_B],False)
            optimizer_G.zero_grad()

            pred_fake_A = D_A(fake_A)
            loss_G_B2A = adv_loss(pred_fake_A, torch.tensor(1.0).expand_as(pred_fake_A).to(device))
            
            pred_fake_B = D_B(fake_B)
            loss_G_A2B = adv_loss(pred_fake_B, torch.tensor(1.0).expand_as(pred_fake_B).to(device))

            loss_cycle_A = cycle_loss(rec_A, real_A)
            loss_cycle_B = cycle_loss(rec_B, real_B)

            if args.lambda_identity>0:
                loss_identity_A = identity_loss(iden_A,real_A)
                loss_identity_B = identity_loss(iden_B,real_B)
                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*args.lambda_cycle + loss_cycle_B*args.lambda_cycle + loss_identity_A*args.lambda_cycle*args.lambda_identity + loss_identity_B*args.lambda_cycle*args.lambda_identity

            else:
                loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A*args.lambda_cycle + loss_cycle_B*args.lambda_cycle

            loss_G.backward()
            optimizer_G.step()

            losses[0]+=loss_G_A2B.item()
            losses[1]+=loss_G_B2A.item()
            losses[2]+=loss_cycle_A.item()
            losses[3]+=loss_cycle_B.item()


            #train discriminator
            set_requires_grad([D_A,D_B],True)
            optimizer_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            fake_A_ = fake_A_buffer.get_images(fake_A)
            pred_fake_A = D_A(fake_A_.detach())
            loss_D_A_real = adv_loss(pred_real_A, torch.tensor(1.0).expand_as(pred_real_A).to(device))
            loss_D_A_fake = adv_loss(pred_fake_A, torch.tensor(0.0).expand_as(pred_fake_A).to(device))
            loss_D_A = (loss_D_A_fake + loss_D_A_real)*0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            fake_B_ = fake_B_buffer.get_images(fake_B)
            pred_fake_B = D_B(fake_B_.detach())
            loss_D_B_real = adv_loss(pred_real_B, torch.tensor(1.0).expand_as(pred_real_B).to(device))
            loss_D_B_fake = adv_loss(pred_fake_B, torch.tensor(0.0).expand_as(pred_fake_B).to(device))
            loss_D_B = (loss_D_B_fake + loss_D_B_real)*0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            losses[4]+=loss_D_A.item() 
            losses[5]+=loss_D_B.item()

            #get sample
            if (epoch * len(train_loader) + i)%args.sample_frequecy ==0:
                images_sample = torch.cat((real_A.data, fake_B.data, rec_A.data, real_B.data, fake_A.data, rec_B.data),0)
                if not os.path.exists("sample/" + args.model):
                    os.makedirs("sample/" + args.model)
                save_image(images_sample, "sample/" + args.model + "/" + str(epoch * len(train_loader) + i) + ".png", nrow=3, normalize=True)
                
    
            current_batch = epoch * len(train_loader) + i
            sys.stdout.write(f"\r[Epoch {epoch+1}/200] [Index {i}/{len(train_loader)}] [D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] [G loss: adv: {loss_G.item():.4f}] [lr: {scheduler_G.get_lr()}]")
            
        
        
        #get tensorboard logs
        if not os.path.exists(args.log_dir + args.model):
            os.makedirs(args.log_dir + args.model)
        writer = SummaryWriter(args.log_dir + args.model)
        writer.add_scalar('loss_G_A2B', losses[0]/float(len(train_loader)), epoch)
        writer.add_scalar('loss_D_A', losses[4]/float(len(train_loader)), epoch)
        writer.add_scalar('loss_G_B2A', losses[1]/float(len(train_loader)), epoch)
        writer.add_scalar('loss_D_B', losses[5]/float(len(train_loader)), epoch)
        writer.add_scalar('loss_cycle_A', losses[2]/float(len(train_loader)), epoch)
        writer.add_scalar('loss_cycle_B', losses[3]/float(len(train_loader)), epoch)
        writer.add_scalar('learning_rate_G', np.array(scheduler_G.get_lr()), epoch)
        writer.add_scalar('learning_rate_D_A', np.array(scheduler_D_A.get_lr()), epoch)
        writer.add_scalar('learning_rate_D_B', np.array(scheduler_D_B.get_lr()), epoch)
        sys.stdout.write(f"[Epoch {epoch+1}/200] [D_A loss: {losses[4]/float(len(train_loader)):.4f}] [D_B loss: {losses[5]/float(len(train_loader)):.4f}] [G adv loss: adv: {losses[0]/float(len(train_loader))+losses[1]/float(len(train_loader)):.4f}]")
        
        #update learning rate
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        if (epoch+1) % args.checkpoint_frequecy == 0:
            if not os.path.exists("models/"+args.model+"/G_A2B/"):
                os.makedirs("models/"+args.model+"/G_A2B/")
            if not os.path.exists("models/"+args.model+"/G_B2A/"):
                os.makedirs("models/"+args.model+"/G_B2A/")
            if not os.path.exists("models/"+args.model+"/D_A/"):
                os.makedirs("models/"+args.model+"/D_A/")
            if not os.path.exists("models/"+args.model+"/D_B/"):
                os.makedirs("models/"+args.model+"/D_B/")
            torch.save(G_A2B.state_dict(), "models/"+args.model+"/G_A2B/"+str(epoch)+".pth")
            torch.save(G_B2A.state_dict(), "models/"+args.model+"/G_B2A/"+str(epoch)+".pth")
            torch.save(D_A.state_dict(), "models/"+args.model+"/D_A/"+str(epoch)+".pth")
            torch.save(D_B.state_dict(), "models/"+args.model+"/D_B/"+str(epoch)+".pth")




if __name__ == "__main__":
    main()