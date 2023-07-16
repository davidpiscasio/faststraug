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
            return transforms.ToTensor()(img).to('cuda')

        if not torch.is_tensor(img):
            img = transforms.ToTensor()(img).to('cuda')
        
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        #img = transforms.ToTensor()(img).to('cuda')
        means = torch.mean(img, dim=(1,2), keepdim=True)
        img = torch.clip((img - means) * c + means, 0, 1)
        return img
    
class Brightness:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            n_channels = img.shape[0]
            img = cp.squeeze(rearrange(cp.asarray(img), 'c h w -> h w c'))
        else:
            n_channels = len(img.getbands())
            img = cp.asarray(img) / 255.
            
        isgray = n_channels == 1

        #img = cp.asarray(img) / 255.
        if isgray:
            img = cp.expand_dims(img, axis=2)
            img = cp.repeat(img, 3, axis=2)

        img = cc.skimage.color.rgb2hsv(img)
        img[:, :, 2] = cp.clip(img[:, :, 2] + c, 0, 1)
        img = cc.skimage.color.hsv2rgb(img)

        img = cp.clip(img, 0, 1)
        img = rearrange(img, 'h w c -> c h w')
        img = torch.as_tensor(img, device='cuda')
        
        if img.shape[0] != 1 and isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img

class Pixelate:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            h, w = img.shape[1:]
        else:
            w, h = img.size
            img = transforms.ToTensor()(img).to('cuda')
        #img = transforms.ToTensor()(img).to('cuda')

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
            return transforms.ToTensor()(img).to('cuda')

        isgray = False
        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        
        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)

        if img.mode != "RGB":
            isgray = True
            img = img.convert("RGB")

        nj = NvJpeg()
        img = np.asarray(img)
        imbytes = nj.encode(img, c)
        img = nj.decode(imbytes)
        img = transforms.ToTensor()(img).to('cuda')

        if isgray:
            img = transforms.functional.rgb_to_grayscale(img)

        return img
