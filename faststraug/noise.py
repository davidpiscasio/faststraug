import torch
import torchvision.transforms as transforms
import cucim as cc
import cupy as cp
import numpy as np
from einops import rearrange

class GaussianNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            n_channels = img.shape[0]
            #img = transforms.ToPILImage()(img)
            img = cp.squeeze(rearrange(cp.asarray(img), 'c h w -> h w c'))
        else:
            n_channels = len(img.getbands())
            img = cp.asarray(img) / 255.
            
        isgray = n_channels == 1

        b = [.06, 0.09, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 0.03)
        
        #img = cp.asarray(img) / 255.
        img = cp.clip(img + cp.random.normal(size=img.shape, scale=c), 0, 1)

        if isgray and img.ndim == 2:
            img = cp.expand_dims(img, axis=0)
        else:
            img = rearrange(img, 'h w c -> c h w')
            
        img = torch.as_tensor(img, device='cuda')
            
        if img.shape[0] != 1 and isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img

class ShotNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            n_channels = img.shape[0]
            #img = transforms.ToPILImage()(img)
            img = cp.squeeze(rearrange(cp.asarray(img), 'c h w -> h w c'))
        else:
            n_channels = len(img.getbands())
            img = cp.asarray(img) / 255.
        isgray = n_channels == 1

        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + 7)
        #img = cp.asarray(img) / 255.
        img = cp.clip(cp.random.poisson(img * c) / float(c), 0, 1)

        if isgray and img.ndim == 2:
            img = cp.expand_dims(img, axis=0)
        else:
            img = rearrange(img, 'h w c -> c h w')
            
        img = torch.as_tensor(img, device='cuda')
            
        if img.shape[0] != 1 and isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img

class ImpulseNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            n_channels = img.shape[0]
            #img = transforms.ToPILImage()(img)
            img = cp.squeeze(rearrange(cp.asarray(img), 'c h w -> h w c'))
        else:
            n_channels = len(img.getbands())
            img = cp.asarray(img) / 255.
        isgray = n_channels == 1

        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .04)
        s = np.random.randint(2 ** 32, size=4)
        img = cc.skimage.util.random_noise(img, mode='s&p', seed=s, amount=c)

        if isgray and img.ndim == 2:
            img = cp.expand_dims(img, axis=0)
        else:
            img = rearrange(img, 'h w c -> c h w')
            
        img = torch.as_tensor(img, device='cuda')
            
        if img.shape[0] != 1 and isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img
        
class SpeckleNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            n_channels = img.shape[0]
            #img = transforms.ToPILImage()(img)
            img = cp.squeeze(rearrange(cp.asarray(img), 'c h w -> h w c'))
        else:
            n_channels = len(img.getbands())
            img = cp.asarray(img) / 255.
        isgray = n_channels == 1

        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = np.random.uniform(a, a + .05)
        #img = cp.asarray(img) / 255.
        img = cp.clip(img + img * cp.random.normal(size=img.shape, scale=c), 0, 1)

        if isgray and img.ndim == 2:
            img = cp.expand_dims(img, axis=0)
        else:
            img = rearrange(img, 'h w c -> c h w')
            
        img = torch.as_tensor(img, device='cuda')
            
        if img.shape[0] != 1 and isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img
