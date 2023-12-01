"""This script creates a table of similarity score from a log file."""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from pprint import pprint as pp

# from matplotlib.axes import Axes

# Light colors for the dots
swarmplot_palette = {
    "Sqa_par": "#8f96bf",
    "Sqa_bif": "#ebb0e5",
    "Sqa_zz": "#9feed3",
}

# Dark colors for the violin
violin_palette = {
    "Sqa_par": "#333c70",
    "Sqa_bif": "#90367c",
    "Sqa_zz": "#34906c",
}

parser = argparse.ArgumentParser(
    prog="Box plotter",
    usage=None,
    description="Table creator",
    add_help=True,
    allow_abbrev=True,
)

parser.add_argument(
    "--table-location",
    "-tl",
    type=str,
    required=True,
    help="Path to the table file.",
)

opts = parser.parse_args()


class BoxPlotCreator:
    """Create a box plot from a txt file with data arranged in a table format."""

    def __init__(self) -> None:
        """Initialize."""
        self.table_location: str = opts.table_location
        self.table: pd.DataFrame = pd.read_csv(self.table_location, sep=",")

    def __score_range(self) -> None:
        """Calculate the range of scores."""
        match self.table["Score name"][0]:
            case "msssim":
                self.score_range: np.ndarray = np.arange(0, 1, 0.1)
            case "psnr":
                self.score_range: np.ndarray = np.arange(0, 100, 10)
            case "psnrb":
                self.score_range: np.ndarray = np.arange(0, 100, 10)
            case "sam":
                self.score_range: np.ndarray = np.arange(0, 180, 18)
            case _:
                raise ValueError("Score name not recognized.")

    def execute(self) -> None:
        """Execute pipeline."""
        self.__score_range()
        self.__gen_swarmplot()

    def __gen_swarmplot(self) -> None:
        """Generate a violin boxplot."""
        # init settings for plot
        sns.set_context(context="paper")
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

        # create violin plot
        sns.violinplot(
            data=self.table,
            ax=axs[0],
            x="Score name",
            y="Score value",
            inner="box",  # alternatives are: box, point, stick
            linewidth=1.5,
        )

        # create swarm plot
        sns.swarmplot(
            data=self.table,
            ax=axs[0],
            x="Score name",
            y="Score value",
            color="white",
            edgecolor="gray",
            s=8,  # Circle size
        )

        # create box plot
        sns.boxplot(
            data=self.table,
            ax=axs[1],
            x="Score name",
            y="Score value",
            linewidth=1.5,
        )

        # overlap stripplot
        sns.stripplot(
            data=self.table,
            ax=axs[1],
            x="Score name",
            y="Score value",
            color="orange",
            jitter=True,
            size=5,
        )

        # Add horizontal grid
        for ax, xlb in zip(axs, ["Violin plot", "Box plot"]):
            ax.set_xlabel(xlb)
            ax.set_ylabel("Score value")
            ax.grid(axis="y")
            ax.set_axisbelow(True)

        plt.ylim(
            self.table["Score value"].min()
            - 0.2 * self.table["Score value"].max(),
            self.table["Score value"].max()
            + 0.2 * self.table["Score value"].max(),
        )

        fig.suptitle(f"Similarity score : {self.table['Score name'][0]}")

        match = re.search(r"\d(?=/)", self.table_location)
        if match:
            plt.savefig(fname=f"./{match.group()}/plots.png")
        else:
            raise ValueError("No match found.")


if __name__ == "__main__":
    box_plot_creator: BoxPlotCreator = BoxPlotCreator()
    box_plot_creator.execute()
