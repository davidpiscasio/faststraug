import numpy as np
import cupy as cp
import math
import torch
import kornia
import torchvision.transforms as transforms
from einops import rearrange
from .ops import plasma_fractal

class Fog:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if mag < 0 or mag >= len(c):
            index = cp.random.randint(0, len(c)).get()
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = cp.asarray(img) / 255.
        max_val = img.max()
        # Make sure fog image is at least twice the size of the input image
        max_size = 2 ** math.ceil(math.log2(max(w, h)) + 1)
        fog = c[0] * plasma_fractal(mapsize=max_size, wibbledecay=c[1])[:h, :w][..., cp.newaxis]
        #print(type(img))
        #print(type(fog))
        # x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        # return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        if isgray:
            fog = cp.squeeze(fog)
        else:
            fog = cp.repeat(fog, 3, axis=2)

        img += fog
        img = cp.clip(img * max_val / (max_val + c[0]), 0, 1)

        img = rearrange(img, 'h w c -> c h w')
        return torch.as_tensor(img, device='cuda')
        #return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))

class Snow:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size

        c = [(0.1, 0.3, 3, 0.5, 5, 0.5, 0.8),
             (0.2, 0.3, 2, 0.5, 7, 0.5, 0.7),
             (0.55, 0.3, 4, 0.9, 11, 0.5, 0.7)]
        
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = transforms.ToTensor()(img).to('cuda')

        '''
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        '''
        
        snow_layer = torch.normal(size=(h,w), mean=c[0], std=c[1])  # [:2] for monochrome

        snow_layer[snow_layer < c[3]] = 0

        snow_layer = snow_layer.to('cuda')

        snow_layer = torch.clip(snow_layer, 0, 1)
        snow_layer = kornia.filters.motion_blur(torch.unsqueeze(torch.unsqueeze(snow_layer, 0), 0), kernel_size=c[4], angle=np.random.uniform(-135,-45), direction=c[5])
        snow_layer = torch.squeeze(snow_layer, 0)

        img = c[6] * img
        gray_img = (1 - c[6]) * torch.maximum(img, kornia.color.rgb_to_grayscale(img, rgb_weights=torch.tensor([0.299, 0.587, 0.114])) * 1.5 + 0.5)
        img += gray_img
        img = torch.clip(img + snow_layer + torch.rot90(snow_layer, k=2, dims=[1,2]), 0, 1)

        '''
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)
        '''
        return img
        
class Frost:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = transforms.functional.pil_to_tensor(img).to('cuda')

        c = [(0.78, 0.22), (0.64, 0.36), (0.5, 0.5)]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]

        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')]

        index = np.random.randint(0, len(filename))
        filename = filename[index]
        # Some images have transparency. Remove alpha channel.
        frost = transforms.functional.pil_to_tensor(Image.open(filename).convert('RGB')).to('cuda')

        # Resize the frost image to match the input image's dimensions
        f_h, f_w = frost.shape[1:]
        if w / h > f_w / f_h:
            f_h = round(f_h * w / f_w)
            f_w = w
        else:
            f_w = round(f_w * h / f_h)
            f_h = h
        frost = transforms.Resize(size=(f_h, f_w), antialias=True)(frost)

        # randomly crop
        y_start, x_start = np.random.randint(0, f_h - h + 1), np.random.randint(0, f_w - w + 1)
        frost = transforms.functional.crop(frost, y_start, x_start, h, w)

        #n_channels = len(img.getbands())
        n_channels = img.shape[0]
        isgray = n_channels == 1

        '''
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        '''
        img = torch.clip(torch.round(c[0] * img + c[1] * frost), 0, 255)

        '''
        if isgray:
            img = ImageOps.grayscale(img)
        '''
        return img / 255.
        
class Rain:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        max_length = min(w, h, 10)
        c = [50, 70, 90]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]
        
        img = transforms.ToTensor()(img).to('cuda')
        rain = kornia.augmentation.RandomRain(p=prob, number_of_drops=(c, c + 20), drop_height=(5, max_length), drop_width=(1,2))
        return torch.squeeze(rain(img))
