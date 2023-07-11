"""This script converts .mat files to .png files for the purpose of visualizing the data in the .mat files."""
# Tested on Python 3.10
import re
import sys

import matplotlib.pyplot as plt
import scipy.io as sio
from utils import *

if __name__ == "__main__":
    # Get the arguments passed to the script
    # The first argument is the path to the directory containing the .mat files
    # The second argument is the path to the directory where the .mat files will be moved
    args = read_arguments(len(sys.argv))

    logger = logger_setup()

    # Convert the .mat files to .png files and save them in the directory passed as an argument to the script
    logger.info(f"Converting {args[1]} to .png file...")
    mat = sio.loadmat(args[1])
    for i in range(
        mat[
            list(filter(lambda x: not x.startswith("__"), mat.keys()))[0]
        ].shape[3]
    ):
        printProgressBar(
            i + 1,
            mat[
                list(filter(lambda x: not x.startswith("__"), mat.keys()))[0]
            ].shape[3],
            prefix="Progress:",
            suffix="Complete",
        )

        plt.imsave(
            f"{re.search(r'.*(?=/convex.*)', args[1]).group()}/{re.search(r'convex.*(?=.mat)', args[1]).group()}_frame-{i+1}.png",
            mat[list(filter(lambda x: not x.startswith("__"), mat.keys()))[0]][
                :, :, :, i
            ],
        )
    logger.info(f"Convertion done!")
