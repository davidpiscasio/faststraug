import cupy as cp
import math
from PIL import Image
from .ops import plasma_fractal

class Fog:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = cp.asarray(img) / 255.
        max_val = img.max()
        # Make sure fog image is at least twice the size of the input image
        max_size = 2 ** math.ceil(math.log2(max(w, h)) + 1)
        fog = c[0] * plasma_fractal(mapsize=max_size, wibbledecay=c[1], rng=self.rng)[:h, :w][..., cp.newaxis]
        #print(type(img))
        #print(type(fog))
        # x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        # return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        if isgray:
            fog = cp.squeeze(fog)
        else:
            fog = cp.repeat(fog, 3, axis=2)

        img += fog
        img = cp.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return Image.fromarray(cp.asnumpy(img).astype(cp.uint8))