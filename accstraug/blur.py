import torchvision.transforms as transforms
import cupy as cp

class GaussianBlur:
    def __call__(self, img, mag=-1, prob=1.):
        if cp.random.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # kernel = [(31,31)] prev 1 level only
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if mag < 0 or mag >= len(sigmas):
            index = self.rng.integers(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]

        img = transforms.functional.pil_to_tensor(img)
        img = img.to('cuda')
        img = transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)
        return transforms.ToPILImage()(img)
