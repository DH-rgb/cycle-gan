import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, instance_norm=True, relu=True, relu_slope=None, init_zero_weights=False):
    """
    畳み込み層を積み上げる。識別ネットワークや生成ネットワークの構成で使う
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    else:
        nn.init.normal_(conv_layer.weight.data, 0.0, 0.02)
    layers.append(conv_layer)

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        if relu_slope:
            relu_layer = nn.LeakyReLU(relu_slope, True)
        else:
            relu_layer = nn.ReLU(inplace=True)
        layers.append(relu_layer)
    return layers

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=1, instance_norm=True, relu=True, relu_slope=None, init_zero_weights=False):

    layers = []
    deconv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)
    if init_zero_weights:
        deconv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    else:
        nn.init.normal_(deconv_layer.weight.data, 0.0, 0.02)
    layers.append(deconv_layer)

    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))

    if relu:
        if relu_slope:
            relu_layer = nn.LeakyReLU(relu_slope, True)
        else:
            relu_layer = nn.ReLU(inplace=True)
        layers.append(relu_layer)
    return layers

class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()

        conv_layers = [
                nn.ReflectionPad2d(1),
                *conv(input_features, input_features, kernel_size=3, stride=1, padding=0),
                nn.ReflectionPad2d(1),
                *conv(input_features, input_features, kernel_size=3, stride=1, padding=0, relu=False)
            ]
        self.model = nn.Sequential(*conv_layers)

    def forward(self, input_data):
        return input_data + self.model(input_data)

class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
        super(Generator, self).__init__()

        # First 7 x 7 convolutional layer
        layers = [
            nn.ReflectionPad2d(3),
            *conv(in_channels, 64, 7, stride=1, padding=0)
        ]

        # Two 3 x 3 convolutional layers
        input_features = 64
        output_features = input_features * 2
        for _ in range(2):
            layers += conv(input_features, output_features, 3)
            input_features, output_features = output_features, output_features * 2

        # Residual blocks
        for _ in range(res_blocks):
            layers += [ResidualBlock(input_features)]

        # Two 3 x 3 deconvolutional layers
        output_features = input_features // 2
        for _ in range(2):
            layers += deconv(input_features, output_features, 3)
            input_features, output_features = output_features, output_features // 2

        # Output layer
        layers += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_features, out_channels, 7),
                nn.Tanh()
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, real_image):
        return self.model(real_image)

class Discriminator(nn.Module):

    def __init__(self, in_channels=3, conv_dim=64):
        super(Discriminator, self).__init__()

        C64 = conv(in_channels, conv_dim, instance_norm=False, relu_slope=0.2)
        C128 = conv(conv_dim, conv_dim * 2, relu_slope=0.2)
        C256 = conv(conv_dim * 2, conv_dim * 4, relu_slope=0.2)
        C512 = conv(conv_dim * 4, conv_dim * 8, stride = 1, relu_slope=0.2)
        C1 = conv(conv_dim * 8, 1, stride=1, instance_norm=False, relu=False)

        self.model = nn.Sequential(
                *C64,
                *C128,
                *C256,
                *C512,
                *C1
            )

    def forward(self, image):
        return self.model(image)