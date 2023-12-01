"""This script creates a table of similarity score from a log file."""
import argparse
import re

import pandas as pd
import plotly.figure_factory as ff
from tabulate import tabulate

parser = argparse.ArgumentParser(
    prog="Table creator",
    usage=None,
    description='Table creator',
    add_help=True,
    allow_abbrev=True,
    )

parser.add_argument(
    '--log-location',
    '-ll',
    type=str,
    required=True,
    help='Path to the log file.',
    )

opts = parser.parse_args()

class TableCreator():
    """Creates a table of similarity score from a log file."""

    def __init__(self, log_location: str=opts.log_location):
        """Initialize."""
        self.log_location = log_location
        self.sev_score = ""

    def create_table(self):
        """Create table of similarity score from a log file."""
        self.table = []
        self.sev_score = re.search(r"\d{1}(?=\.log)", self.log_location)
        if self.sev_score:
            self.sev_score = self.sev_score.group(0)

        with open(self.log_location, 'r') as log_file:
            for line in log_file:
                score_name = re.search(r"(?<=->\s)\S+", line)
                score_value = re.search(r"(?<=:\s).*", line)
                if score_name and score_value:
                    score_name = score_name.group(0)
                    score_value = score_value.group(0)
                imgs = re.findall(r"synthetic\S+", line)
                if len(imgs) > 0 :
                    self.table.append([imgs[1], imgs[0], score_name, score_value])

    def write_table(self):
        """Write the table to a file."""
        with open(f"./results/Evaluation/{self.sev_score}/{self.sev_score}.txt", "w") as table_file:
            table_file.write(
                    tabulate(
                    tabular_data=self.table,
                    headers=['Image 1', 'Image 2', 'Score name', 'Score value'],
                    tablefmt='fancy_outline')
                )

            # table_file.write('\n\n')

            # table_file.write(
            #         tabulate(
            #         tabular_data=self.table,
            #         headers=['Image 1', 'Image 2', 'Score name', 'Score value'],
            #         tablefmt='html')
            #     )

    def save_as_df(self):
        """Save the table as a pandas dataframe."""
        df = pd.DataFrame(self.table[:35], columns=['Image 1', 'Image 2', 'Score name', 'Score value'])
        print(df)
        fig = ff.create_table(df)
        fig.update_layout(
            autosize=True,
            )
        df.to_csv(f"./results/Evaluation/{self.sev_score}/{self.sev_score}.csv", index=False)
        fig.write_image(f"./results/Evaluation/{self.sev_score}/{self.sev_score}.png")

if __name__ == '__main__':
    table_creator = TableCreator()
    table_creator.create_table()
    table_creator.write_table()
    table_creator.save_as_df()
