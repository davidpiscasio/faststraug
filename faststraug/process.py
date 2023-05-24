import numpy as np
import torchvision.transforms as transforms

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
        img = transforms.functional.pil_to_tensor(img).to('cuda')
        img = transforms.functional.posterize(img, bit)
        return img
    
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
        img = transforms.functional.pil_to_tensor(img).to('cuda')
        return transforms.functional.solarize(img, thresh)
    
class Invert:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        return transforms.functional.invert(transforms.functional.pil_to_tensor(img).to('cuda'))

class Sharpness:
    def __call__(self, img, mag=-1, prob=1.):
        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c + .6)
        
        return transforms.RandomAdjustSharpness(magnitude, p=prob)(transforms.ToTensor()(img).to('cuda'))
    
class Equalize:
    def __call__(self, img, mag=-1, prob=1.):
        return transforms.RandomEqualize(p=prob)(transforms.functional.pil_to_tensor(img).to('cuda'))

class AutoContrast:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        return transforms.functional.autocontrast(transforms.functional.pil_to_tensor(img).to('cuda'))
    
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

        return transforms.functional.adjust_saturation(transforms.functional.pil_to_tensor(img).to('cuda'), magnitude)