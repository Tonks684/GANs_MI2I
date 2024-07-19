import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import sys

from torchvision import models

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    """
    Initialize the weights of the given module.

    Args:
        m (nn.Module): The module to initialize the weights for.

    Returns:
        None
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    """
    Returns the normalization layer based on the given norm_type.

    Parameters:
    norm_type (str): The type of normalization layer to be used. Options are 'batch' and 'instance'.

    Returns:
    norm_layer (function): The normalization layer function based on the norm_type.

    Raises:
    NotImplementedError: If the given norm_type is not supported.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, use_dropout=False, norm='instance', gpu_ids=[]):
    """
    Defines the generator network for the pix2pixHD model.

    Args:
        input_nc (int): Number of input channels.
        output_nc (int): Number of output channels.
        ngf (int): Number of generator filters in the first conv layer.
        netG (str): Type of generator network. Can be 'global', 'local', or 'encoder'.
        n_downsample_global (int): Number of downsampling layers in the global generator.
        n_blocks_global (int): Number of residual blocks in the global generator.
        n_local_enhancers (int): Number of local enhancers in the local generator.
        n_blocks_local (int): Number of residual blocks in each local enhancer.
        use_dropout (bool): Whether to use dropout layers in the generator.
        norm (str): Type of normalization layer. Can be 'instance' or 'batch'.
        gpu_ids (list): List of GPU IDs to use.

    Returns:
        torch.nn.Module: The generator network.

    Raises:
        Exception: If the specified generator network is not implemented.
    """

    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, use_dropout, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, use_dropout, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise Exception('Generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    """
    Define the discriminator network.

    Args:
        input_nc (int): Number of input channels.
        ndf (int): Number of discriminator filters in the first convolutional layer.
        n_layers_D (int): Number of layers in the discriminator.
        norm (str, optional): Type of normalization layer. Defaults to 'instance'.
        use_sigmoid (bool, optional): Whether to use a sigmoid activation function. Defaults to False.
        num_D (int, optional): Number of discriminators to use. Defaults to 1.
        getIntermFeat (bool, optional): Whether to return intermediate features. Defaults to False.
        gpu_ids (list, optional): List of GPU IDs to use. Defaults to [].

    Returns:
        torch.nn.Module: Discriminator network.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    """
    GANLoss is a class that defines the adversarial loss for a GAN (Generative Adversarial Network).

    Args:
        use_lsgan (bool): Whether to use least squares GAN loss (MSE loss) or binary cross-entropy GAN loss (BCE loss).
        target_real_label (float): The target label value for real samples.
        target_fake_label (float): The target label value for fake samples.
        tensor (torch.Tensor): The tensor type to use for creating label tensors.

    Attributes:
        real_label (float): The target label value for real samples.
        fake_label (float): The target label value for fake samples.
        real_label_var (torch.Tensor): The variable storing the real label tensor.
        fake_label_var (torch.Tensor): The variable storing the fake label tensor.
        Tensor (torch.Tensor): The tensor type to use for creating label tensors.
        loss (nn.Module): The loss function to use for calculating the GAN loss.

    Methods:
        get_target_tensor(input, target_is_real): Returns the target tensor based on the input and target_is_real flag.
        __call__(input, target_is_real): Computes the GAN loss given the input and target_is_real flag.

    """

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        """
        Returns the target tensor based on the input and target_is_real flag.

        Args:
            input (torch.Tensor): The input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            torch.Tensor: The target tensor.

        """
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        """
        Computes the GAN loss given the input and target_is_real flag.

        Args:
            input (torch.Tensor): The input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            torch.Tensor: The computed GAN loss.

        """
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    """
    Calculates the VGG loss between two input tensors.

    Args:
        gpu_ids (list): List of GPU IDs to use for computation.

    Attributes:
        vgg (Vgg19): VGG19 model used for feature extraction.
        criterion (nn.L1Loss): L1 loss criterion used for pixel-wise comparison.
        weights (list): List of weights for each VGG feature map.

    """

    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        """
        Calculates the VGG loss between two input tensors.

        Args:
            x (torch.Tensor): Input tensor x.
            y (torch.Tensor): Input tensor y.

        Returns:
            torch.Tensor: VGG loss between x and y.

        """
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    """
    LocalEnhancer module that combines a global generator model with local enhancer layers.

    Args:
        input_nc (int): Number of input channels.
        output_nc (int): Number of output channels.
        ngf (int): Number of filters in the generator.
        n_downsample_global (int): Number of downsampling layers in the global generator.
        n_blocks_global (int): Number of residual blocks in the global generator.
        n_local_enhancers (int): Number of local enhancer layers.
        n_blocks_local (int): Number of residual blocks in each local enhancer layer.
        use_dropout (bool): Whether to use dropout layers.
        norm_layer (nn.Module): Normalization layer to use.
        padding_type (str): Type of padding to use.

    Returns:
        torch.Tensor: Output tensor.

    """

    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, use_dropout=False, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, use_dropout, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, use_dropout=use_dropout, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        """
        Forward pass of the LocalEnhancer module.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    """
    Global Generator network for pix2pixHD model.

    Args:
        input_nc (int): Number of input channels.
        output_nc (int): Number of output channels.
        ngf (int): Number of filters in the generator's first conv layer. Default is 64.
        n_downsampling (int): Number of downsampling layers. Default is 3.
        n_blocks (int): Number of residual blocks in the generator. Default is 9.
        use_dropout (bool): Whether to use dropout layers. Default is True.
        norm_layer (nn.Module): Normalization layer to use. Default is nn.BatchNorm2d.
        padding_type (str): Type of padding. Default is 'reflect'.

    Attributes:
        model (nn.Sequential): Sequential model representing the generator network.

    """

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, use_dropout=True, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, use_dropout=use_dropout, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        Forward pass of the Global Generator network.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.model(input)

class ResnetBlock(nn.Module):
    """
    Residual block implementation for the ResNet architecture.

    Args:
        dim (int): Number of input and output channels.
        padding_type (str): Type of padding to be applied. Options are 'reflect', 'replicate', or 'zero'.
        norm_layer (nn.Module): Normalization layer to be used.
        use_dropout (bool): Whether to use dropout or not. Default is False.
        activation (nn.Module): Activation function to be used. Default is nn.ReLU(True).

    Returns:
        torch.Tensor: Output tensor of the residual block.

    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout=False, activation=nn.ReLU(True)):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        """
        Build the convolutional block for the residual block.

        Args:
            dim (int): Number of input and output channels.
            padding_type (str): Type of padding to be applied. Options are 'reflect', 'replicate', or 'zero'.
            norm_layer (nn.Module): Normalization layer to be used.
            activation (nn.Module): Activation function to be used.
            use_dropout (bool): Whether to use dropout or not.

        Returns:
            nn.Sequential: Sequential module containing the convolutional block.

        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]
        else:
            conv_block += [nn.Dropout(0)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the residual block.

        """
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    """
    Encoder network for pix2pixHD model.

    Args:
        input_nc (int): Number of input channels.
        output_nc (int): Number of output channels.
        ngf (int): Number of filters in the first layer. Default is 32.
        n_downsampling (int): Number of downsampling layers. Default is 4.
        norm_layer (nn.Module): Normalization layer. Default is nn.BatchNorm2d.

    Attributes:
        output_nc (int): Number of output channels.
        model (nn.Sequential): Sequential model containing the encoder layers.

    """

    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        """
        Forward pass of the encoder network.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_nc, H, W).
            inst (torch.Tensor): Instance tensor of shape (batch_size, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_nc, H, W).

        """
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    """
    Multiscale Discriminator network for image-to-image translation.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        """
        Initialize the MultiscaleDiscriminator.

        Args:
            input_nc (int): Number of input channels.
            ndf (int): Number of discriminator filters in the first convolutional layer.
            n_layers (int): Number of layers in each discriminator.
            norm_layer (nn.Module): Normalization layer.
            use_sigmoid (bool): Whether to use a sigmoid activation function.
            num_D (int): Number of discriminators to use.
            getIntermFeat (bool): Whether to get intermediate features from each discriminator.
        """
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        """
        Forward pass through a single discriminator.

        Args:
            model (nn.Module): Discriminator model.
            input (torch.Tensor): Input tensor.

        Returns:
            list: List of intermediate features if `getIntermFeat` is True, otherwise a list with a single element.
        """
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        """
        Forward pass through the multiscale discriminator.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            list: List of discriminator outputs at different scales.
        """
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    """
    A class representing the N-Layer PatchGANDiscriminator network.

    Args:
        input_nc (int): Number of input channels.
        ndf (int, optional): Number of discriminator filters in the first layer. Default is 64.
        n_layers (int, optional): Number of layers in the discriminator. Default is 3.
        norm_layer (nn.Module, optional): Normalization layer. Default is nn.BatchNorm2d.
        use_sigmoid (bool, optional): Whether to use a sigmoid activation function. Default is False.
        getIntermFeat (bool, optional): Whether to return intermediate features. Default is False.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        """
        Forward pass of the discriminator network.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class Vgg19(torch.nn.Module):
    """
    Vgg19 model for feature extraction.

    Args:
        requires_grad (bool): Whether to require gradients for the model parameters. Default is False.

    Attributes:
        slice1 (torch.nn.Sequential): Sequential module for the first slice of Vgg19.
        slice2 (torch.nn.Sequential): Sequential module for the second slice of Vgg19.
        slice3 (torch.nn.Sequential): Sequential module for the third slice of Vgg19.
        slice4 (torch.nn.Sequential): Sequential module for the fourth slice of Vgg19.
        slice5 (torch.nn.Sequential): Sequential module for the fifth slice of Vgg19.

    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        """
        Forward pass of the Vgg19 model.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            list: List of feature maps extracted at different layers of Vgg19.

        """
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
