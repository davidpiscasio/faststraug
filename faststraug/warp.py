import cv2
import numpy as np
from PIL import Image, ImageOps
import kornia
import torch
import torchvision.transforms as transforms

class Stretch:
    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = transforms.ToTensor()(img).to('cuda')
        img = torch.unsqueeze(img,0)

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        srcpt = np.zeros((1,10,2))
        dstpt = np.zeros((1,10,2))

        # left-most
        srcpt[0,0,:] = [p, p]
        srcpt[0,1,:] = [p, h - p]
        srcpt[0,2,:] = [p, h_50]
        x = np.random.uniform(0, frac) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt[0,0,:] = [p + x, p]
        dstpt[0,1,:] = [p + x, h - p]
        dstpt[0,2,:] = [p + x, h_50]

        # 2nd left-most
        srcpt[0,3,:] = [p + w_33, p]
        srcpt[0,4,:] = [p + w_33, h - p]
        x = np.random.uniform(-frac, frac) * w_33
        dstpt[0,3,:] = [p + w_33 + x, p]
        dstpt[0,4,:] = [p + w_33 + x, h - p]

        # 3rd left-most
        srcpt[0,5,:] = [p + w_66, p]
        srcpt[0,6,:] = [p + w_66, h - p]
        x = np.random.uniform(-frac, frac) * w_33
        dstpt[0,5,:] = [p + w_66 + x, p]
        dstpt[0,6,:] = [p + w_66 + x, h - p]

        # right-most
        srcpt[0,7,:] = [w - p, p]
        srcpt[0,8,:] = [w - p, h - p]
        srcpt[0,9,:] = [w - p, h_50]
        x = np.random.uniform(-frac, 0) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt[0,7,:] = [w - p + x, p]
        dstpt[0,8,:] = [w - p + x, h - p]
        dstpt[0,9,:] = [w - p + x, h_50]

        srcpt = torch.from_numpy(srcpt).to('cuda')
        dstpt = torch.from_numpy(dstpt).to('cuda')

        srcpt[:,:,0] = 2.0*(srcpt[:,:,0]/w) - 1.0
        dstpt[:,:,0] = 2.0*(dstpt[:,:,0]/w) - 1.0
        srcpt[:,:,1] = 2.0*(srcpt[:,:,1]/h) - 1.0
        dstpt[:,:,1] = 2.0*(dstpt[:,:,1]/h) - 1.0

        kernel_weights, affine_weights = kornia.geometry.transform.get_tps_transform(dstpt, srcpt)
        img = kornia.geometry.transform.warp_image_tps(img.double(), srcpt, kernel_weights, affine_weights, align_corners=True)
        return torch.squeeze(img)
    
class Distort:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = transforms.ToTensor()(img).to('cuda')
        img = torch.unsqueeze(img,0)

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        srcpt = np.zeros((1,8,2))
        dstpt = np.zeros((1,8,2))

        # top pts
        srcpt[0,0,:] = [p, p]
        x = np.random.uniform(0, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt[0,0,:] = [p + x, p + y]

        srcpt[0,1,:] = [p + w_33, p]
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt[0,1,:] = [p + w_33 + x, p + y]

        srcpt[0,2,:] = [p + w_66, p]
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt[0,2,:] = [p + w_66 + x, p + y]

        srcpt[0,3,:] = [w - p, p]
        x = np.random.uniform(-frac, 0) * w_33
        y = np.random.uniform(0, frac) * h_50
        dstpt[0,3,:] = [w - p + x, p + y]

        # bottom pts
        srcpt[0,4,:] = [p, h - p]
        x = np.random.uniform(0, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt[0,4,:] = [p + x, h - p + y]

        srcpt[0,5,:] = [p + w_33, h - p]
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt[0,5,:] = [p + w_33 + x, h - p + y]

        srcpt[0,6,:] = [p + w_66, h - p]
        x = np.random.uniform(-frac, frac) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt[0,6,:] = [p + w_66 + x, h - p + y]

        srcpt[0,7,:] = [w - p, h - p]
        x = np.random.uniform(-frac, 0) * w_33
        y = np.random.uniform(-frac, 0) * h_50
        dstpt[0,7,:] = [w - p + x, h - p + y]

        srcpt = torch.from_numpy(srcpt).to('cuda')
        dstpt = torch.from_numpy(dstpt).to('cuda')

        srcpt[:,:,0] = 2.0*(srcpt[:,:,0]/w) - 1.0
        dstpt[:,:,0] = 2.0*(dstpt[:,:,0]/w) - 1.0
        srcpt[:,:,1] = 2.0*(srcpt[:,:,1]/h) - 1.0
        dstpt[:,:,1] = 2.0*(dstpt[:,:,1]/h) - 1.0

        kernel_weights, affine_weights = kornia.geometry.transform.get_tps_transform(dstpt, srcpt)
        img = kornia.geometry.transform.warp_image_tps(img.double(), srcpt, kernel_weights, affine_weights, align_corners=True)
        return torch.squeeze(img)
    
class Curve:
    def __init__(self, square_side=224, rng=None):
        self.side = square_side

    def __call__(self, img, mag=-1, prob=1.):
        if np.random.uniform(0, 1) > prob:
            return img

        orig_w, orig_h = img.size

        img = transforms.ToTensor()(img).to('cuda')

        if orig_h != self.side or orig_w != self.side:
            img = transforms.Resize(size=(self.side, self.side), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(img)

        isflip = np.random.uniform(0, 1) > 0.5
        if isflip:
            #img = ImageOps.flip(img)
            img = transforms.functional.vflip(img)

        img = torch.unsqueeze(img,0)

        w = self.side
        h = self.side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w

        b = [1.1, .95, .8]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag

        rmin = b[index]

        r = np.random.uniform(rmin, rmin + .1) * h
        x1 = (r ** 2 - w_50 ** 2) ** 0.5
        h1 = r - x1

        t = np.random.uniform(0.4, 0.5) * h

        w2 = w_50 * t / r
        hi = x1 * t / r
        h2 = h1 + hi

        sinb_2 = ((1 - x1 / r) / 2) ** 0.5
        cosb_2 = ((1 + x1 / r) / 2) ** 0.5
        w3 = w_50 - r * sinb_2
        h3 = r - r * cosb_2

        w4 = w_50 - (r - t) * sinb_2
        h4 = r - (r - t) * cosb_2

        w5 = 0.5 * w2
        h5 = h1 + 0.5 * hi
        h_50 = 0.50 * h

        srcpt = np.array([[(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h), (0, h_50), (w, h_50)]])
        dstpt = np.array([[(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4), (w - w4, h4), (w5, h5), (w - w5, h5)]])

        srcpt = torch.from_numpy(srcpt).to('cuda')
        dstpt = torch.from_numpy(dstpt).to('cuda')

        srcpt[:,:,0] = 2.0*(srcpt[:,:,0]/w) - 1.0
        dstpt[:,:,0] = 2.0*(dstpt[:,:,0]/w) - 1.0
        srcpt[:,:,1] = 2.0*(srcpt[:,:,1]/h) - 1.0
        dstpt[:,:,1] = 2.0*(dstpt[:,:,1]/h) - 1.0

        kernel_weights, affine_weights = kornia.geometry.transform.get_tps_transform(dstpt, srcpt)
        img = kornia.geometry.transform.warp_image_tps(img.double(), srcpt, kernel_weights, affine_weights, align_corners=True)
        img = torch.squeeze(img)

        if isflip:
            img = transforms.functional.vflip(img)
            rect = (self.side // 2, 0, self.side // 2, self.side)
        else:
            rect = (0, 0, self.side // 2, self.side)

        img = transforms.functional.resized_crop(img, top=rect[0], left=rect[1], height=rect[2], width=rect[3], size=(orig_h, orig_w), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)

        return img