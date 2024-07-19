import random
import torch
from torch.autograd import Variable
class ImagePool():
    """
    A class representing an image pool for storing and retrieving images.

    Attributes:
        pool_size (int): The maximum size of the image pool.
        num_imgs (int): The current number of images in the pool.
        images (list): A list of images stored in the pool.

    Methods:
        query(images): Retrieves a batch of images from the pool, replacing some of them with the input images.
    """

    def __init__(self, pool_size):
        """
        Initializes an ImagePool object.

        Args:
            pool_size (int): The maximum size of the image pool.
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Retrieves a batch of images from the pool, replacing some of them with the input images.

        Args:
            images (torch.Tensor): The input images to be added to the pool.

        Returns:
            torch.Tensor: The batch of images retrieved from the pool.
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
