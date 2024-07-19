import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random


class BaseDataset(data.Dataset):
    """
    A base class for custom datasets in the pix2pixHD project.
    """

    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        """
        Returns the name of the dataset.
        """
        return 'BaseDataset'

    def initialize(self, opt):
        """
        Initializes the dataset with the given options.

        Args:
            opt (argparse.Namespace): The command line arguments.

        """
        pass

def get_params(opt, img):
    """
    Generate random parameters for data augmentation.

    Args:
        opt (argparse.Namespace): The command line arguments.
        size (tuple): The size of the input image (width, height).

    Returns:
        dict: A dictionary containing the generated parameters.
            - 'crop_pos': A tuple representing the top-left corner of the cropped region.
            - 'flip': A boolean indicating whether to flip the image horizontally.
    """
    w, h = opt.loadSize, opt.loadSize
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=False):
    """
    Returns a composed transformation for image preprocessing.

    Args:
        opt (argparse.Namespace): Options for the transformation.
        params (dict): Parameters for the transformation.
        method (PIL.Image.Resampling, optional): Resampling method for resizing. Defaults to Image.BICUBIC.
        normalize (bool, optional): Whether to normalize the image. Defaults to True.

    Returns:
        torchvision.transforms.Compose: Composed transformation.

    """
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(opt,img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(opt,img, params['crop_pos'], opt.fineSize)))

    if 'center_crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __cent_crop(opt,img, params['crop_pos'], opt.fineSize)))


    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(opt,img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5),
                                                (0.5))]

    return transforms.Compose(transform_list)

def normalize():
    """
    Returns a transformation that normalizes the input image tensor.
    
    The normalization is performed by subtracting the mean (0.5, 0.5, 0.5) and dividing by the standard deviation (0.5, 0.5, 0.5) for each channel.
    
    Returns:
        transforms.Normalize: A transformation that normalizes the input image tensor.
    """
    return transforms.Normalize((0.5), (0.5))

def __make_power_2(opt, img, base, method=Image.BICUBIC):
    """
    Resizes the input image to the nearest power of 2 dimensions.

    Args:
        opt (object): The options object.
        img (PIL.Image): The input image.
        base (int): The base value for rounding the dimensions.
        method (int, optional): The resampling method. Defaults to Image.BICUBIC.

    Returns:
        PIL.Image: The resized image.
    """
    ow, oh = opt.loadSize, opt.loadSize
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(opt, img, target_width, method=Image.BICUBIC):
    """
    Scale the width of an image to a target width while maintaining the aspect ratio.

    Args:
        opt (object): The options object.
        img (PIL.Image.Image): The input image.
        target_width (int): The target width to scale the image to.
        method (int, optional): The resampling method to use. Defaults to Image.BICUBIC.

    Returns:
        PIL.Image.Image: The scaled image.
    """
    ow, oh = opt.loadSize, opt.loadSize
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def __crop(opt, img, pos, size):
    """
    Crop the input image based on the given position and size.

    Parameters:
        opt (object): An object containing options and settings.
        img (PIL.Image): The input image to be cropped.
        pos (tuple): The position (x, y) of the top-left corner of the crop.
        size (int): The size of the crop (width and height).

    Returns:
        PIL.Image: The cropped image.
    """
    ow, oh = opt.loadSize, opt.loadSize
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __cent_crop(opt, img, pos, size):
    """
    Crop the input image to the specified size, centered at the image's center.

    Args:
        opt (object): An object containing options and settings.
        img (PIL.Image): The input image to be cropped.
        pos (tuple): The position of the image.
        size (int): The desired size of the cropped image.

    Returns:
        PIL.Image: The cropped image.
    """
    ow, oh = opt.loadSize, opt.loadSize
    start_x = (ow - size) // 2
    start_y = (oh - size) // 2
    end_y = oh - start_y
    end_x = ow - start_x

    return img.crop((start_x, start_y, end_x, end_y))
    

def __flip(img, flip):
    """
    Flip the image horizontally if flip is True.

    Parameters:
    img (PIL.Image.Image): The input image.
    flip (bool): Whether to flip the image or not.

    Returns:
    PIL.Image.Image: The flipped image if flip is True, otherwise the original image.
    """
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
