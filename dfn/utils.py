"""
Utility functions
"""

import numpy as np


def filters_image(filter_data):
    """
    Combine filters for one data instance in a single image

    Parameters:
    -----------
    filter_data : np.ndarray
        filter data of shape (fs, fs, image_rows, image_cols)
    """

    fs = filter_data.shape[0]
    rows, cols = filter_data.shape[2:]
    image = np.zeros((fs * rows, fs * cols))

    for i in range(rows):
        for j in range(cols):
            image[fs * i:fs * (i + 1), fs * j:fs * (j + 1)] = filter_data[:, :,
                                                                          i, j]
    return image


def get_filter(model, x):
    """
    Return filter values for given input
    """

    gen = model.layers[1].gen

    f = gen.get_output(x).eval()
    s = int(np.sqrt(f.shape[1]))
    new_shape = [f.shape[0], s, s, f.shape[-2], f.shape[-1]]
    return np.reshape(f, new_shape)
