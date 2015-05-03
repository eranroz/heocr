__author__ = 'eran'
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.signal import convolve2d

def down_sample(img_data):
    img_data = img_data[:img_data.shape[0]-img_data.shape[0] % 2, :img_data.shape[1]-img_data.shape[1] % 2, :]
    img_data = img_data[::2, ::2, :]
    return img_data


def grad_graph(img, interactive=True):
    """
    gradient graph: use gradient to define different objects and then segment the image to those objects
    :param img: image to process
    :return: fragmentation
    """

    blur_radius = 2
    imgf = ndimage.gaussian_filter(img, blur_radius)


    scharr2 = np.array([[ -3, 0,  3],
                       [-10, 0, 10],
                       [ -3, 0,  3]])
    grad = [convolve2d(imgf[:, :, i], scharr2, boundary='symm', mode='same') for i in np.arange(3)]
    grad = np.sum(np.abs(grad), 0, dtype=np.uint8)
    imgf[ndimage.gaussian_filter(grad/np.mean(grad), 3) < 1] = 0
    threshold = np.mean(imgf)
    mask = np.any(imgf > threshold, -1)
    labeled, nr_objects = ndimage.label(mask)
    # labeled: a width X height matrix, in which each pixel of object i is given index i

    sizes = ndimage.sum(mask, labeled, range(nr_objects + 1))
    #plt.title('objects are colored')
    #plt.imshow(labeled)
    #plt.show()
    labeled[(sizes < 1000)[labeled]] = 0  # filter noise
    label_im = np.searchsorted(np.unique(labeled), labeled)
    # TODO: can we do better with widths and heights (Eran: I think it is the way to calc width/height)
    widths_heights = np.abs(np.diff((np.array(ndimage.extrema(mask, label_im, range(np.max(label_im) + 1))[2:])), axis=0)[0, :, :])
    label_im[(widths_heights[:, 0] < 5) | (widths_heights[:, 0] > 800) | (widths_heights[:, 1] > 300)] = 0
    label_im = np.searchsorted(np.unique(label_im), label_im)

    #plt.hist(widths_heights[:, 0],100) width histogram
    #plt.show()

    # now given the objects return their positions (frags)
    frags = ndimage.find_objects(label_im, max_label=np.max(label_im))
    if interactive:
        # show example of one of the words
        img2 = np.array(img, copy=True)
        img2[label_im == 0] = 0
        plt.imshow(img2)
        plt.show()

        #
        fragment_to_show = len(frags)//2
        plt.imshow(img[frags[fragment_to_show]])
        plt.show()

    return frags


def word_segmentation(img_data):
    segments = grad_graph(img_data)
    return segments

