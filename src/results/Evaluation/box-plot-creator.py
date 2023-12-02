"""This script creates a table of similarity score from a log file."""
import argparse
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from pprint import pprint as pp

# from matplotlib.axes import Axes

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
                # filter from inf values
                self.table["Score value"].replace(
                    to_replace=np.inf, value=10000, inplace=True
                )
                self.table.dropna(
                    subset=["Score value"], how="all", inplace=True
                )
            case "psnrb":
                self.score_range: np.ndarray = np.arange(0, 100, 10)
                # filter from inf values
                self.table["Score value"].replace(
                    to_replace=[np.inf, -np.inf], value=np.nan, inplace=True
                )
                self.table.dropna(
                    subset=["Score value"], how="all", inplace=True
                )
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
        palette = sns.color_palette("mako").as_hex()
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))

        # create violin plot
        axs[0] = sns.violinplot(
            data=self.table,
            ax=axs[0],
            x="Score name",
            y="Score value",
            inner="box",  # alternatives are: box, point, stick
            linewidth=1.5,
            hue="Score name",
            palette=["#7209B7"],
            legend=False,
        )

        # create swarm plot
        axs[0] = sns.swarmplot(
            data=self.table,
            ax=axs[0],
            x="Score name",
            y="Score value",
            color="yellow",
            edgecolor="auto",
            s=8,  # Circle size
        )

        # create box plot
        axs[1] = sns.boxplot(
            data=self.table,
            ax=axs[1],
            x="Score name",
            y="Score value",
            linewidth=1.5,
            hue="Score name",
            palette=["#4361EE"],
            legend=False,
        )

        # overlap stripplot
        axs[1] = sns.stripplot(
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

        fig.suptitle(
            # NOTE: This is a hacky way to get the score name since in case of psnr and psnrb
            # a line might be dropped due to inf values and the first line might not be present.
            f"Similarity score : {self.table['Score name'][self.table.index.values[0]]}"
        )

        # Set transparency and hatch for violin plot
        for patch in axs[0].collections[::2]:
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)
            patch.set_alpha(0.4)
            patch.set_hatch("///")

        # Set transparency and hatch for box plot
        for patch in axs[1].patches:
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)
            patch.set_alpha(0.4)
            patch.set_hatch("+++")

        match = re.search(r"\d(?=/)", self.table_location)
        if match:
            plt.savefig(fname=f"./{match.group()}/plots.png")
        else:
            raise ValueError("No match found.")


if __name__ == "__main__":
    box_plot_creator: BoxPlotCreator = BoxPlotCreator()
    box_plot_creator.execute()
