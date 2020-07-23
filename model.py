import torch 
import torch.nn as nn 
import torch.nn.functional as F


def init_weights(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(net.weight.data, 1.0, 0.02)
        torch.nn.init.constant(net.bias.data, 0.0)



class ResidualBlock(nn.Module):
    """Some Information about ResidualBlock"""
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,kernel_size=3,stride=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,256,kernel_size=3,stride=1),
            nn.InstanceNorm2d(256)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x



class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self,img_channel,res_block):
        super(Generator, self).__init__()
        self.encode_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(img_channel,64,kernel_size=7,stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128,256,kernel_size=3,stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        res_blocks = [ResidualBlock() for _ in range(res_block)]
        self.res_block = nn.Sequential(
            *res_blocks
        )
        self.decode_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.ConvTranspose2d(64,img_channel,kernel_size=7,stride=1),
            nn.InstanceNorm2d(img_channel),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.encode_block(x)
        x = self.res_block(x)
        x = self.decode_block(x)
        return x


class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self,img_channel):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(img_channel,64,kernel_size=4,stride=2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64,128,kernel_size=4,stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128,256,kernel_size=4,stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256,512,kernel_size=4,stride=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.conv_last = nn.Conv2d(512,1,kernel_size=4,stride=1)


    def forward(self, x):
        x = self.block(x)
        x = self.conv_last(x)
        return x



