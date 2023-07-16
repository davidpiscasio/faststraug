import torchvision.transforms as transforms
import numpy as np
import torch
import kornia

class Shrink:
    def __init__(self, rng=None):
        self.translateXAbs = TranslateXAbs()
        self.translateYAbs = TranslateYAbs()

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim < 3 else img
            h, w = img.shape[1:]
        else:
            w, h = img.size
            img = transforms.ToTensor()(img).to('cuda')
        #img = transforms.ToTensor()(img).to('cuda')
        img = torch.unsqueeze(img,0)

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        srcpt = np.zeros((1,8,2))
        dstpt = np.zeros((1,8,2))

        # left-most
        srcpt[0,0,:] = [p, p]
        srcpt[0,1,:] = [p, h - p]
        x = np.random.uniform(frac - .1, frac) * w_33
        y = np.random.uniform(frac - .1, frac) * h_50
        dstpt[0,0,:] = [p + x, p + y]
        dstpt[0,1,:] = [p + x, h - p - y]

        # 2nd left-most 
        srcpt[0,2,:] = [p + w_33, p]
        srcpt[0,3,:] = [p + w_33, h - p]
        dstpt[0,2,:] = [p + w_33, p + y]
        dstpt[0,3,:] = [p + w_33, h - p - y]

        # 3rd left-most 
        srcpt[0,4,:] = [p + w_66, p]
        srcpt[0,5,:] = [p + w_66, h - p]
        dstpt[0,4,:] = [p + w_66, p + y]
        dstpt[0,5,:] = [p + w_66, h - p - y]

        # right-most 
        srcpt[0,6,:] = [w - p, p]
        srcpt[0,7,:] = [w - p, h - p]
        dstpt[0,6,:] = [w - p - x, p + y]
        dstpt[0,7,:] = [w - p - x, h - p - y]

        srcpt = torch.from_numpy(srcpt).to('cuda')
        dstpt = torch.from_numpy(dstpt).to('cuda')

        srcpt[:,:,0] = 2.0*(srcpt[:,:,0]/w) - 1.0
        dstpt[:,:,0] = 2.0*(dstpt[:,:,0]/w) - 1.0
        srcpt[:,:,1] = 2.0*(srcpt[:,:,1]/h) - 1.0
        dstpt[:,:,1] = 2.0*(dstpt[:,:,1]/h) - 1.0

        kernel_weights, affine_weights = kornia.geometry.transform.get_tps_transform(dstpt, srcpt)
        img = kornia.geometry.transform.warp_image_tps(img.double(), srcpt, kernel_weights, affine_weights, align_corners=True)

        if np.random.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return torch.squeeze(img, 0)

class Perspective:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return transforms.ToTensor()(img).to('cuda')

        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim < 3 else img
            h, w = img.shape[1:]
        else:
            w, h = img.size
            img = transforms.ToTensor()(img).to('cuda')
        #img = transforms.ToTensor()(img).to('cuda')
        src = [[0, 0], [w, 0], [0, h], [w, h]]

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
            return transforms.ToTensor()(img).to('cuda')

        #w, h = img.size
        if torch.is_tensor(img):
            img = torch.unsqueeze(img, 0) if img.ndim < 3 else img
            h, w = img.shape[1:]
            img = img*255
            img = img.to(torch.uint8)
        else:
            w, h = img.size
            img = transforms.functional.pil_to_tensor(img).to('cuda')
        #img = transforms.functional.pil_to_tensor(img).to('cuda')
        
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

        return img / 255.
    
class TranslateXAbs:
    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0, 1) > 0.5:
            v = -v

        m = torch.Tensor([[[1, 0, v],[0, 1, 0]]]).to('cuda')
        return kornia.geometry.transform.warp_affine(img, M=m.double(), dsize=img.shape[2:])

class TranslateYAbs:
    def __call__(self, img, val=0, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        v = np.random.uniform(0, val)

        if np.random.uniform(0, 1) > 0.5:
            v = -v

        m = torch.Tensor([[[1, 0, 0],[0, 1, v]]]).to('cuda')
        return kornia.geometry.transform.warp_affine(img, M=m.double(), dsize=img.shape[2:])
