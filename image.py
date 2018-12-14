from torchvision import transforms
from PIL import Image
import numpy as np


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class Rescale():
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        return resize_and_place_image_to_shape(img, self.scale)


def resize_and_place_image_to_shape(img, shape):
    oh = img.height
    ow = img.width

    rh = shape[1] / oh
    rw = shape[0] / ow

    ratio = min(rh, rw)
    nw = int(round(ow*ratio))
    nh = int(round(oh*ratio))
    sized = img.resize((nw, nh), resample=Image.BICUBIC)
    assert (sized.size[0] <= shape[0] and sized.size[1] <= shape[1])

    nx = (shape[0] - nw) // 2
    ny = (shape[1] - nh) // 2

    canvas = Image.new('RGB', shape, (127, 127, 127))
    canvas.paste(sized, (nx, ny))

    return canvas


class RandomColorDistort():
    def __init__(self, hue=0.2, saturation=1.2, exposure=1.2):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, img):
        dhue = np.random.uniform(-self.hue, self.hue)
        dsat = rand_scale(self.saturation)
        dexp = rand_scale(self.exposure)
        res = self.distort_image(img, dhue, dsat, dexp)
        return res

    @staticmethod
    def distort_image(im, hue, sat, val):
        im = im.convert('HSV')
        cs = list(im.split())
        cs[1] = cs[1].point(lambda i: i * sat)
        cs[2] = cs[2].point(lambda i: i * val)

        def change_hue(x):
            x += hue * 255
            if x > 255:
                x -= 255
            if x < 0:
                x += 255
            return x

        cs[0] = cs[0].point(change_hue)
        im = Image.merge(im.mode, tuple(cs))

        im = im.convert('RGB')
        return im


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(2) == 0:
        return scale
    return 1./scale

