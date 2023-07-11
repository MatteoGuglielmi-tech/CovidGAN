"""Launch this script to plot the distribution of the gravity scores."""
import sys
from os import listdir
from os.path import isfile
from os.path import join
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_score_distribution(scores: Dict) -> None:
    """Plot the distribution of the gravity scores.

    :param scores: the dictionary containing the scores
    """
    options_scores = list(scores.keys())
    density_scores = list(scores.values())

    fig = plt.figure(figsize=(10, 5))
    plt.bar(
        options_scores,
        density_scores,
        color=[
            "yellowgreen",
            "yellow",
            "orange",
            "red",
        ],
        width=0.4,
    )

    plt.xlabel("Possible gravity scores")
    plt.ylabel("No. of images per score")
    plt.yticks(np.arange(0, 14750, step=500))
    plt.grid(axis="y", alpha=0.75)
    plt.title("Images distribution per gravity score")
    plt.show()

    fig.savefig("./score_distribution.png")


if __name__ == "__main__":
    # 2 because the first argument is the script name
    arguments = [sys.argv[i] for i in range(2)]
    scores = {}
    for i in range(0, 4):
        path = arguments[1] + str(i)
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        scores[str(i)] = len(onlyfiles)

    plot_score_distribution(scores)
