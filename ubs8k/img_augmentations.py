import numpy as np
import skimage.filters as filters
import skimage.transform as transform
from PIL import Image
from skimage import exposure

from .augmentations import ImgAugmentation


def random_interpolation():
    """
    Returns:
        a random interpolation filter for the Image library
    """

    filters = [Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC]
    return np.random.choice(filters)


class Equalize(ImgAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return exposure.equalize_hist(data)
    

class Frangi(ImgAugmentation):
    # TODO add more parameters
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return -1 * filters.frangi(data)


class Gabor(ImgAugmentation):
    def __init__(self, ratio, frequency: float = 0.5):
        super().__init__(ratio)
        self.frequency = frequency
        
    def apply_helper(self, data):
        results_real, result_imag = filters.gabor(data, self.frequency)
        return results_real

    
class Gaussian(ImgAugmentation):
    def __init__(self, ratio, sigma):
        super().__init__(ratio)
        self.sigma = sigma
        
    def apply_helper(self, data):
        return filters.gaussian(data, sigma=self.sigma)
    
    
class Hessian(ImgAugmentation):
    def __init__(self, ratio, sigmas = [1], mode = 'reflect'):
        super().__init__(ratio)
        self.sigmas = sigmas
        self.mode = mode
        
    def apply_helper(self, data):
        return filters.hessian(data, sigmas=self.sigmas)


class Meijering(ImgAugmentation):
    def __init__(self, ratio, sigmas = [1], mode = 'reflect'):
        super().__init__(ratio)
        self.sigmas = sigmas
        self.mode = mode
        
    def apply_helper(self, data):
        return filters.meijering(data, sigmas=self.sigmas)

    
class Prewitt(ImgAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return filters.prewitt(data)
    

class Sobel(ImgAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return filters.sobel(data)


class Roberts(ImgAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return filters.roberts(data)
    
    

class Roberts(ImgAugmentation):
    def __init__(self, ratio):
        super().__init__(ratio)
        
    def apply_helper(self, data):
        return filters.roberts(data)
    

class Transform(ImgAugmentation):
    def __init__(self, ratio, scale=(1, 1), rotation=(0, 0), translation=(0, 0)):
        super().__init__(ratio)
        self.scale = scale
        self.rotation = rotation
        self.translation = translation
        
    def apply_helper(self, data):
        scale = np.random.uniform(*self.scale)
        rotation = np.random.uniform(*self.rotation)
        translation = np.random.randint(*self.translation) if self.translation != (0,0) else 0
        
        tform = transform.SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
        return transform.warp(data, tform)
    
# class Equalize(Augmentation):
#     def __init__(self, ratio):
#         super().__init__(ratio)
        
#     def _apply(self, data):
#         return exposure.equalize_hist(data)