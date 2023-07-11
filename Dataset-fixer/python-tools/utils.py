"""Utilities functions."""
import logging
import os
import sys
from typing import List
from typing import Tuple

import imageio.v3 as iio
import numpy as np


def printProgressBar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
) -> None:
    """Call in a loop to create terminal progress bar.

    @params:
        iteration   - Required  : current
        total       - Required  : total iterations
        prefix      - Optional  : prefix string
        suffix      - Optional  : suffix string
        decimals    - Optional  : positive number of decimals in percent complete
        length      - Optional  : character length of print
        fill        - Optional  : bar fill character
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(
        " \r % s | % s | % s % s % s " % (prefix, bar, percent, "%", suffix),
        end=" \r ",
    )
    # Print New Line on Complete
    if iteration == total:
        print()


def check_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if two numpy arrays are equalself.

    :param a: the first array
    :param b: the second array
    :return: True if the arrays are equal, False otherwise
    """
    return np.array_equal(a, b)


def read_arguments(howmany: int) -> List:
    """Read the arguments passed to the script.

    :return: a list containing the arguments passed to the script
    """
    return [sys.argv[i] for i in range(howmany)]


def logger_setup() -> logging.Logger:
    """Set the logger.

    :return: the logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def extractFolderContent(folder: str) -> List:
    """Extract the content of a folder.

    :param folder: the folder to extract the content from
    :return: a list containing the content of the folder
    """
    paths = []
    for path, _, files in os.walk(folder):
        for file in files:
            paths.append(os.path.join(path, file))

    return paths


def biggestImageDimensions(files: List, path: str) -> Tuple:
    """Find the biggest image dimensions in a list of images.

    :param files: the list of images
    :param path: the path to the folder where the images reside
    :return: a tuple containing the biggest image dimensions
    """
    height = {}
    width = {}
    for i, file in enumerate(files):
        image = iio.imread(os.path.join(path, file))
        height[file] = image.shape[0]
        width[file] = image.shape[1]
        printProgressBar(
            iteration=i + 1,
            total=len(files),
            prefix="Progress:",
            suffix="Complete",
            length=100,
        )
    max_height_idx = list(height.values()).index(max(list(height.values())))
    max_width_idx = list(width.values()).index(max(list(width.values())))
    max_height = max(list(height.values()))
    max_width = max(list(width.values()))
    print(
        f"Max height: {max_height} -> Given by: {list(height.keys())[max_height_idx]}"
    )
    print(
        f"Max width: {max_width} -> Given by: {list(width.keys())[max_width_idx]}"
    )
    return max_height, max_width


def padding(files: List, path: str, finalDims: Tuple) -> None:
    """Pad the given image to the specified dimensions.

    :param files: the list of images
    :param path: the path to the folder where the images reside
    :param finalDims: the final dimensions of the images
    """
    for i, file in enumerate(files):
        image = iio.imread(os.path.join(path, file))
        h, w = image.shape[:2]
        if h == finalDims[0] and w == finalDims[1]:
            continue
        else:
            padH = finalDims[0] - h
            padW = finalDims[1] - w
            pad = ((0, padH), (0, padW), (0, 0))
            padded = np.pad(image, pad, mode="constant", constant_values=(0, 0))
            iio.imwrite(os.path.join(path, file), padded)
            printProgressBar(
                iteration=i + 1,
                total=len(files),
                prefix="Progress:",
                suffix="Complete",
                length=100,
            )
