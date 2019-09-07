import numpy as np
import time
from PIL import Image
from config import IMG_H,IMG_W

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
    im = im.resize((IMG_W,IMG_H),Image.ANTIALIAS)
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
