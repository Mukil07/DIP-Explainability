from PIL import Image
import random


## frame level augmentation 

class DriverFocusCrop(object):
    """Randomly crop the area where the driver is
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, scales, size, interpolation=Image.BILINEAR):

        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        image_width = img.size[0]
        image_height = img.size[1]

        img = img.crop((self.tl_x, self.tl_y, image_width-self.tl_x1, image_height-self.tl_y1))

        return img

    def randomize_parameters(self):

        self.tl_x = random.randint(200, 400)
        self.tl_y = random.randint(0, 100)
        self.tl_x1 = random.randint(200, 400)
        self.tl_y1 = random.randint(0, 100)

class horizontal_flip(object):
    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
#        if self.p < 0.5:
#            return img.transpose(Image.FLIP_LEFT_RIGHT)
#        return img
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    def randomize_parameters(self):
        self.p = random.random()  
      
