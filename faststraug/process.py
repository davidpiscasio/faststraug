import numpy as np
import torchvision.transforms as transforms
import torch

class Posterize:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        c = [6, 3, 1]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c)).get()
        else:
            index = mag
        c = c[index]
        bit = np.random.randint(c, c + 2)
        
        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
            
        img = transforms.functional.posterize(img, bit)
        return img / 255.
    
class Solarize:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        c = [192, 128, 64]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c)).get()
        else:
            index = mag
        c = c[index]
        thresh = np.random.randint(c, c + 64)
        
        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        
        return transforms.functional.solarize(img, thresh) / 255.
    
class Invert:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        
        return transforms.functional.invert(img) / 255.

class Sharpness:
    def __call__(self, img, mag=-1, prob=1.):
        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        
        if not torch.is_tensor(img):
            img = transforms.ToTensor()(img).to('cuda')
        
        return transforms.RandomAdjustSharpness(magnitude, p=prob)(img)
    
class Equalize:
    def __call__(self, img, mag=-1, prob=1.):
        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        
        return transforms.RandomEqualize(p=prob)(img) / 255.

class AutoContrast:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        
        return transforms.functional.autocontrast(img) / 255.
    
class Color:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        c = [.1, .5, .9]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim == 2 else img
            img = img*255
            img = img.to(torch.uint8)
        else:
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        
        return transforms.functional.adjust_saturation(img, magnitude) / 255.
