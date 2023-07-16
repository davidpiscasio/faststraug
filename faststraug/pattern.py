import numpy as np
import torchvision.transforms as transforms
from PIL import ImageDraw
import torch

class VGrid:
    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1., isgrid=False):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)
        
        if copy:
            img = img.copy()
        w, h = img.size

        if mag < 0 or mag > max_width:
            line_width = np.random.randint(1, max_width)
            image_stripe = np.random.randint(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = w // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            x = image_stripe * i + line_width * (i - 1)
            draw.line([(x, 0), (x, h)], width=line_width, fill='black')

        if isgrid:
            return img

        return transforms.ToTensor()(img).to('cuda')
    
class HGrid:
    def __call__(self, img, copy=True, max_width=4, mag=-1, prob=1., isgrid=False):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)
        
        if copy:
            img = img.copy()
        w, h = img.size
        if mag < 0 or mag > max_width:
            line_width = np.random.randint(1, max_width)
            image_stripe = np.random.randint(1, max_width)
        else:
            line_width = 1
            image_stripe = 3 - mag

        n_lines = h // (line_width + image_stripe) + 1
        draw = ImageDraw.Draw(img)
        for i in range(1, n_lines):
            y = image_stripe * i + line_width * (i - 1)
            draw.line([(0, y), (w, y)], width=line_width, fill='black')

        if isgrid:
            return img

        return transforms.ToTensor()(img).to('cuda')

class Grid:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)
        
        img = VGrid()(img, copy=True, mag=mag, isgrid=True)
        img = HGrid()(img, copy=False, mag=mag, isgrid=True)
        return transforms.ToTensor()(img).to('cuda')
    
class RectGrid:
    def __call__(self, img, isellipse=False, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)
        
        img = img.copy()
        w, h = img.size
        line_width = 1
        image_stripe = 3 - mag  # self.rng.integers(2, 6)
        offset = 4 if isellipse else 1
        n_lines = ((h // 2) // (line_width + image_stripe)) + offset
        draw = ImageDraw.Draw(img)
        x_center = w // 2
        y_center = h // 2
        for i in range(1, n_lines):
            dx = image_stripe * i + line_width * (i - 1)
            dy = image_stripe * i + line_width * (i - 1)
            x1 = x_center - (dx * w // h)
            y1 = y_center - dy
            x2 = x_center + (dx * w / h)
            y2 = y_center + dy
            if isellipse:
                draw.ellipse([(x1, y1), (x2, y2)], width=line_width, outline='black')
            else:
                draw.rectangle([(x1, y1), (x2, y2)], width=line_width, outline='black')
                
        if isellipse:
            return img

        return transforms.ToTensor()(img).to('cuda')
    
class EllipseGrid:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img)
        
        img = RectGrid()(img, isellipse=True, mag=mag, prob=prob)
        return transforms.ToTensor()(img).to('cuda')
