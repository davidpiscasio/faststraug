from PIL import ImageOps
import torchvision.transforms as transforms
import numpy as np
import cupy as cp
import cucim as cc
import torch
from einops import rearrange
from nvjpeg import NvJpeg

class Contrast:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        img = transforms.ToTensor()(img).to('cuda')
        means = torch.mean(img, dim=(1,2), keepdim=True)
        img = torch.clip((img - means) * c + means, 0, 1)
        return img
    
class Brightness:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        # W, H = img.size
        # c = [.1, .2, .3, .4, .5]
        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = cp.asarray(img) / 255.
        if isgray:
            img = cp.expand_dims(img, axis=2)
            img = cp.repeat(img, 3, axis=2)

        img = cc.skimage.color.rgb2hsv(img)
        img[:, :, 2] = cp.clip(img[:, :, 2] + c, 0, 1)
        img = cc.skimage.color.hsv2rgb(img)

        img = cp.clip(img, 0, 1)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.as_tensor(img, device='cuda')
        if isgray:
            img = ImageOps.grayscale(img)

        return img

class Pixelate:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = transforms.ToTensor()(img).to('cuda')
        # c = [0.6, 0.5, 0.4, 0.3, 0.25]
        c = [0.6, 0.5, 0.4]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
            
        c = c[index]
        img = transforms.functional.resize(img, (int(h * c), int(w * c)), interpolation=transforms.InterpolationMode.NEAREST)
        return transforms.functional.resize(img, (h, w), interpolation=transforms.InterpolationMode.NEAREST)

class JpegCompression:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        nj = NvJpeg()
        img = np.asarray(img)
        imbytes = nj.encode(img, c)
        img = nj.decode(imbytes)
        return transforms.ToTensor()(img).to('cuda')