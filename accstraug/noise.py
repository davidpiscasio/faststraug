import cupy as cp
import cucim as cc
from PIL import Image

class GaussianNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        b = [.06, 0.09, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = cp.random.uniform(a, a + 0.03)
        img = cp.asarray(img) / 255.
        img = cp.clip(img + cp.random.normal(size=img.shape, scale=c.get()), 0, 1) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))

class AccShotNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(3, 60)
        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = cp.random.uniform(a, a + 7)
        img = cp.asarray(img) / 255.
        img = cp.clip(cp.random.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))

class ImpulseNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = cp.random.uniform(a, a + .04)
        s = cp.random.random_integers(2 ** 32, size=4)
        img = cc.skimage.util.random_noise(cp.asarray(img) / 255., mode='s&p', seed=s.get(), amount=c.get()) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))

class SpeckleNoise:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = cp.random.uniform(a, a + .05)
        img = cp.asarray(img) / 255.
        img = cp.clip(img + img * cp.random.normal(size=img.shape, scale=c.get()), 0, 1) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))