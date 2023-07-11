"""This script allows to apply a binary mask to an image.

Assumptions:
    - The mask is a binary image (.png)
    - The mask is in a folder called "Binary Mask" inside the folder containing all the scroes folders
    - The mask has the same name of the original video with a "_mask" postfix
    - The mask has the same dimensions as the resized frame
"""
import os
import re

import cv2 as cv
import numpy as np
from utils import *

bck_value = 0

if __name__ == "__main__":
    # getting arguments
    script_name, imageFullPath = read_arguments(
        len(sys.argv)
    )  # ex. imageFullPath = ../../Dati-San-Matteo-Dataset-2/0/convex_1017_1047_10_frame-100.png
    print(f"imageFullPath: {imageFullPath}")
    # setting up logger
    logger = logger_setup()
    # getting info from the path
    videoName = re.search(r"convex.*(?=_frame-\d+)", imageFullPath).group(
        0
    )  # ex. convex_1017_1047_10
    frameName = re.search(r"convex.*(?=.png)", imageFullPath).group(
        0
    )  # ex. convex_1017_1047_10_frame-100
    path2frame = re.search(r".*(?=/convex)", imageFullPath).group(
        0
    )  # ex. ../../Dati-San-Matteo-Dataset-2/0
    path2BinaryMaskFolder = (
        f"{re.search(r'.*(?=/[0-9]{1}/convex)', imageFullPath).group(0)}"
    )

    if "Binary-Masks" in imageFullPath or "masked" in imageFullPath:
        logger.info(f"The image is already masked : {frameName}")
        sys.exit(0)
    else:
        # load image
        logger.info(f"Loading {videoName}...")
        img = cv.imread(imageFullPath)

        # load mask
        logger.info(
            f"Loading {path2BinaryMaskFolder}/Binary-Masks/{videoName}_mask.png"
        )
        mask = cv.imread(
            f"{path2BinaryMaskFolder}/Binary-Masks/{videoName}_mask.png",
            cv.IMREAD_GRAYSCALE,
        )
        # convert to B&W
        # mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv.threshold(
            mask, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
        )

        hh, ww, _ = img.shape
        # getting contours of masked portion.
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            x, y, w, h = cv.boundingRect(c)

        startx = (ww - w) // 2
        starty = (hh - h) // 2
        result = np.zeros_like(img)
        result[starty : starty + h, startx : startx + w] = img[y : y + h, x : x + w]

        # saving image
        logger.info(f"Saving {frameName} centered in {path2frame}/{frameName}...")
        cv.imwrite(f"{path2frame}/{frameName}_masked.png", result)
        logger.info(f"Removing {frameName}... ")
        os.system(f"rm {imageFullPath}")
