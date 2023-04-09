import cupy as cp
from cucim.skimage.filters import gaussian
import numpy as np
import torch
import torchvision.transforms as transforms
import kornia
from einops import rearrange
from cupyx.scipy.signal import correlate2d
from PIL import ImageOps
from .ops import disk

class GaussianBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if mag < 0 or mag >= len(sigmas):
            index = cp.random.randint(0, len(sigmas)).get()
        else:
            index = mag
        sigma = sigmas[index]
        img = transforms.ToTensor()(img)
        img = img.to('cuda')
        img = transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)
        return img

class ZoomBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        img = transforms.ToTensor()(img).to('cuda')
        h, w = img.shape[1:]
        c = [np.arange(1, 1.11, .01),
             np.arange(1, 1.16, .01),
             np.arange(1, 1.21, .02)]
        if mag < 0 or mag >= len(c):
            index = cp.random.randint(0, len(c))
        else:
            index = mag

        c = c[index]

        n_channels = img.shape[0]
        isgray = n_channels == 1

        out = torch.zeros_like(img, device='cuda')
        for zoom_factor in c:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)
            zoom_img = transforms.functional.resize(img, size=(zh, zw), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            x1 = (zw - w) // 2
            y1 = (zh - h) // 2
            zoom_img = transforms.functional.crop(zoom_img, y1, x1, h, w)
            out += zoom_img

        img = (img + out) / (len(c) + 1)

        img = torch.clip(img, 0, 1)
        
        return img

class DefocusBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(2, 0.1), (3, 0.1), (4, 0.1)]  # , (6, 0.5)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = np.random.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        img = cp.asarray(img) / 255.
        if isgray:
            img = cp.expand_dims(img, axis=2)
            img = cp.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])
        kernel = cp.asarray(kernel)

        channels = []
        for d in range(n_channels):
            channels.append(correlate2d(img[:, :, d], kernel, boundary='symm', mode='same'))
        channels = cp.asarray(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        img = cp.clip(channels, 0, 1)
        img = rearrange(img, 'h w c -> c h w')
        if isgray:
            img = ImageOps.grayscale(img)

        return torch.as_tensor(img, device='cuda')

class FastMotionBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        #c = [(10, 3), (12, 4), (14, 5)]
        c = [5,7,9]
        angle = np.random.uniform(-45,45)
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        img = transforms.ToTensor()(img).to('cuda')
        motion_blur = kornia.augmentation.RandomMotionBlur(p=prob, kernel_size=c, angle=45., direction=1.0)

        if isgray:
            img = ImageOps.grayscale(img)

        return torch.squeeze(motion_blur(img))
    
class GlassBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        img = transforms.ToTensor()(img).to('cuda')
        h, w = img.shape[1:]
        # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.45, 1, 1), (0.6, 1, 2), (0.75, 1, 2)]  # , (1, 2, 3)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = np.random.integers(0, len(c))
        else:
            index = mag

        c = c[index]

        img = torch.uint8(gaussian(img, sigma=c[0], channel_axis=-1) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[x, y], img[x_prime, y_prime] = img[x_prime, y_prime], img[x, y]

        img = torch.clip(gaussian(img, sigma=c[0], channel_axis=-1), 0, 1) * 255
        return img