"""Script that moves images to the correct folder according to their score."""
import os
import re

import pandas as pd
from utils import *

if __name__ == "__main__":
    arguments = read_arguments(len(sys.argv))
    logger = logger_setup()

    ## arguments = /run/media/matteoguglielmi/HDD-Partition/Dev/Projects/VAEs-for-Ultrasound-Imaging/Dati-San-Matteo-Dataset-2/1017/1047/convex_1017_1047_10_frame-1.png
    frameNumRegex = re.search(
        r"(?<=_frame-)\d+", arguments[1]
    )  # to get frame number
    videoNumRegex = re.search(
        r"\d+(?=_frame)", arguments[1]
    )  # to get frame number
    pathRegex = re.search(
        r".*(?=/convex.*)", arguments[1]
    )  # to get path leading to current file
    rootNameRegex = re.search(
        r"convex.*(?=_frame)", arguments[1]
    )  # to get root name
    datasetFolderRegex = re.search(
        r".*(?=/\d+/\d+)", arguments[1]
    )  # to get dataset folder name (Dati-San-Matteo-2)

    logger.info(f"Reading {arguments[1]}...")
    file = pd.read_csv(
        f"{re.search(r'.*(?=_frame)', arguments[1]).group()}_total_scores.csv",
        header=None,
    ).drop(index=0, axis=0, inplace=False)
    logger.info(f"File {arguments[1]} read!")

    # this check is to avoid errors when the frame number is greater than the number of frames in the video.
    # = because the frames are numbered from 1 to n, not from 0 to n-1
    # > beacuse the last frame is not labelled in many cases. Checkig whether the lenght of the score list is greater than the frame number is a way to check if the frame is labelled or not
    if int(frameNumRegex.group()) > len(file.iloc[0, :]):
        logger.error(f"Skipping...")
        exit(0)
    else:
        framescore = file.iat[0, int(frameNumRegex.group()) - 1]
        logger.info(
            f"Got score for {rootNameRegex.group()}_frame-{frameNumRegex.group()}.png: {framescore}"
        )
    logger.info(
        f"Moving {rootNameRegex.group()}_frame-{frameNumRegex.group()}.png... to {framescore} folder"
    )
    os.system(
        f"mv {pathRegex.group()}/{rootNameRegex.group()}_frame-{frameNumRegex.group()}.png {datasetFolderRegex.group()}/{framescore}/{rootNameRegex.group()}_frame-{frameNumRegex.group()}.png"
    )
