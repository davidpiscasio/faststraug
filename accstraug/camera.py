import cupy as cp
import cucim as cc
from PIL import Image, ImageOps

class Contrast:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = cp.asarray(img) / 255.
        means = cp.mean(img, axis=(0, 1), keepdims=True)
        img = cp.clip((img - means) * c + means, 0, 1) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))

class Brightness:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        # W, H = img.size
        # c = [.1, .2, .3, .4, .5]
        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = cp.random.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = cp.asarray(img) / 255.
        if isgray:
            img = cp.expand_dims(img, axis=2)
            img = cp.repeat(img, 3, axis=2)

        img = cc.skimage.color.rgb2hsv(img)
        img[:, :, 2] = cp.clip(img[:, :, 2] + c, 0, 1)
        img = cc.skimage.color.hsv2rgb(img)

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = cp.clip(img, 0, 1) * 255
        img = Image.fromarray(cp.asnumpy(img).astype(cp.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img
        # if isgray:
        # if isgray:
        #    img = color.rgb2gray(img)

        # return Image.fromarray(img.astype(np.uint8))