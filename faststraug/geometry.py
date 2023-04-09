import torchvision.transforms as transforms
import numpy as np
from einops import rearrange

class Perspective:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = transforms.ToTensor()(img).to('cuda')
        src = [[0, 0], [w, 0], [0, h], [w, h]]
        # low = 0.3

        b = [.05, .1, .15]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if np.random.uniform(0, 1) > 0.5:
            topright_y = np.random.uniform(low, low + .1) * h
            bottomright_y = np.random.uniform(high - .1, high) * h
            dest = [[0, 0], [w, topright_y], [0, h], [w, bottomright_y]]
        else:
            topleft_y = np.random.uniform(low, low + .1) * h
            bottomleft_y = np.random.uniform(high - .1, high) * h
            dest = [[0, topleft_y], [w, 0], [0, bottomleft_y], [w, h]]

        img = transforms.functional.perspective(img, startpoints=src, endpoints=dest)
        return img

class Rotate:
    def __init__(self, square_side=224, rng=None):
        self.side = square_side

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size

        img = transforms.functional.pil_to_tensor(img).to('cuda')
        if h != self.side or w != self.side:
            img = transforms.Resize(size=(self.side, self.side), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)

        b = [15, 30, 45]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = np.random.uniform(rotate_angle - 20, rotate_angle)
        if np.random.uniform(0, 1) < 0.5:
            angle = -angle

        img = transforms.functional.rotate(img, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR, expand=not iscurve)
        img = transforms.Resize(size=(h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)

        return img