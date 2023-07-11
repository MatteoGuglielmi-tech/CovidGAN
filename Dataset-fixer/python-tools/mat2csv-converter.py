"""This scripts converts a .mat file to a .csv file."""
import csv
import re
import sys

import pandas as pd
import scipy.io as sio
from utils import *

# import os

if __name__ == "__main__":
    # Get the arguments passed to the script
    # The first argument is the path to the directory containing the .mat files
    # The second argument is the path to the directory where the .mat files will be moved
    args = read_arguments(len(sys.argv))
    logger = logger_setup()

    # Convert the .mat files to .csv files and save them in the directory passed as an argument to the script
    logger.info(f"Converting {args[1]} to .csv file...")
    mat = sio.loadmat(args[1])
    with open(
        f"{re.search(r'.*(?=/convex.*)', args[1]).group()}/{re.search(r'convex.*(?=.mat)', args[1]).group()}.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter=",", dialect="excel")
        writer.writerows(
            mat[list(filter(lambda x: not x.startswith("__"), mat.keys()))[0]]
        )
        logger.info(f"File {args[1]} converted to .csv file!")

    try:
        df = pd.read_csv(
            f"{re.search(r'.*(?=/convex.*)', args[1]).group()}/{re.search(r'convex.*(?=.mat)', args[1]).group()}.csv",
            sep=",",
            header=None,
        )
        df = df.sum(axis=0).to_frame().T
        df.to_csv(
            f"{re.search(r'.*(?=/convex.*)', args[1]).group()}/{re.search(r'convex.*(?=_score.mat)', args[1]).group()}_total_scores.csv",
            sep=",",
            index=False,
        )
        logger.info(f"Total scores saved to .csv file!")
    except Exception as e:
        logger.error(f"Error while saving the .csv file: {e}")

    logger.info(f"Convertion done!")
