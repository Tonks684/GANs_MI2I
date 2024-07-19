###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff','.tif'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    """
    A custom dataset class for loading images from a folder.

    Args:
        root (str): The root directory path of the image folder.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
        return_paths (bool, optional): If True, returns the image path along with the image. Default is False.
        loader (callable, optional): A function to load an image given its path. Default is default_loader.

    Attributes:
        root (str): The root directory path of the image folder.
        imgs (list): A list of image paths in the folder.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        return_paths (bool): If True, returns the image path along with the image.
        loader (callable): A function to load an image given its path.

    Methods:
        __getitem__(self, index): Retrieves the image and its path (if return_paths is True) at the given index.
        __len__(self): Returns the total number of images in the dataset.

    """

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in: ' + root + '\n'
                               'Supported image extensions are: ' +
                               ','.join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
