import numpy as np
import time
from PIL import Image
from config import IMG_H, IMG_W


def get_seed():
    # In 115.74 years , you dont will get same seed
    # unless you 'python3 run.py' twice in 0.01s
    # if so , agent and env's seed is same
    # but seed of tensorflow will be different
    t = str(time.time()).split('.')
    first = t[0][-7:]
    second = t[1][:2]
    if len(second) < 2:
        second = second + '0' * (2 - len(second))
    return int(first + second)


def gray_scale(im):
    """
    this function is used to convert an image to gray image

    :param im:  np.array with shape  (*,*,3)
    :return:    np.array with shape  (*,*,1)
    """
    im = Image.fromarray(im).convert("L")
    im = np.array(im).astype("uint8")
    return im[:, :, np.newaxis]


def resize(im):
    """
    this function is used to resize an image

    :param im: np.array with shape  (*,*,3)
    :return:   np.array with shape  (IMG_H,IMG_W,3)
    """
    im = Image.fromarray(im.astype(np.uint8))
    im = im.resize((IMG_W, IMG_H), Image.ANTIALIAS)
    im = np.array(im).astype("uint8")
    return im


def resize_gray(im):
    # combine two functions above
    return gray_scale(resize(im))


# test
if __name__ == '__main__':
    a = np.random.randint(0, 256, (240, 256, 3)).astype("uint8")
    a = resize(a)
    assert a.dtype == "uint8"
    assert a.shape == (IMG_H, IMG_W, 3)
    a = gray_scale(a)
    assert a.dtype == "uint8"
    assert a.shape == (IMG_H, IMG_W, 1)
    a = np.random.randint(0, 256, (240, 256, 3)).astype("uint8")
    a = resize_gray(a)
    assert a.dtype == "uint8"
    assert a.shape == (IMG_H, IMG_W, 1)
    print("test         :  OK")
