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
