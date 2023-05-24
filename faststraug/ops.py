import cv2
import numpy as np
import cupy as cp
from scipy.ndimage import zoom as scizoom
import torch

def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        coords = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        coords = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    x, y = np.meshgrid(coords, coords)
    aliased_disk = np.asarray((x ** 2 + y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3, rng=None):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = cp.empty((mapsize, mapsize), dtype=cp.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * cp.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + cp.roll(cornerref, shift=-1, axis=0)
        squareaccum += cp.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + cp.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + cp.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + cp.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + cp.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def alpha_composite(src, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    #src = np.asarray(src)
    #dst = np.asarray(dst)
    #out = np.empty(src.shape, dtype = 'float')
    img = torch.empty(src.shape).to('cuda')
    #alpha = np.index_exp[:, :, 3:]
    #rgb = np.index_exp[:, :, :3]
    src_r = src[:, :3, :, :] * 255.0
    dst_r = dst[:, :3, :, :] * 255.0
    src_a = src[:, 3:, :, :]
    dst_a = dst[:, 3:, :, :]
    img[:, 3:, :, :] = src_a+dst_a*(1-src_a)
    #old_setting = np.seterr(invalid = 'ignore')
    img[:, :3, :, :] = (src_r*src_a + dst_r*dst_a*(1-src_a))/img[:, 3:, :, :]
    #np.seterr(**old_setting)    
    #out[alpha] *= 255
    img = torch.clip(img,0,1)
    # astype('uint8') maps np.nan (and np.inf) to 0
    #out = out.astype('uint8')
    #out = Image.fromarray(out, 'RGBA')
    return img

def alpha_composite(src, dst):
    '''
    Return the alpha composite of src and dst.

    Parameters:
    src -- PIL RGBA Image object
    dst -- PIL RGBA Image object

    The algorithm comes from http://en.wikipedia.org/wiki/Alpha_compositing
    '''
    # http://stackoverflow.com/a/3375291/190597
    # http://stackoverflow.com/a/9166671/190597
    img = torch.empty(src.shape).to('cuda')
    src_r = src[:, :3, :, :] * 255.0
    dst_r = dst[:, :3, :, :] * 255.0
    src_a = src[:, 3:, :, :]
    dst_a = dst[:, 3:, :, :]

    img[:, 3:, :, :] = dst_a+src_a*(1-dst_a)
    img[:, :3, :, :] = (dst_r*dst_a + src_r*src_a*(1-dst_a))/img[:, 3:, :, :]
    img = torch.clip(img,0,255)

    return img/255.0