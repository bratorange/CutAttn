from argparse import ArgumentParser
from pathlib import Path

from subcommand import Subcommand, register_subcommand


@register_subcommand
class VisScores(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("name", type=str)

    @staticmethod
    def invoke(experiments, args):
        from pathlib import Path

        import numpy as np
        from matplotlib import pyplot as plt

        name = args.name

        loaded_data = np.load(Path("results") / name / "scores.npz")
        metric_names = loaded_data['metric_names']
        scores = loaded_data['scores']
        epochs = loaded_data['epochs']

        markers = ['o', 'v', 's', '*']
        colors = ['red', 'blue', 'yellow', 'green']


        for i, (score_name, score, marker, color) in enumerate(zip(metric_names, scores, markers, colors)):
            if i == 0:
                fig, axes = plt.subplots()
                axes.set_xlabel("epoch")
            else:
                axes = axes.twinx()
            axes.plot(epochs, score, marker=marker, color=color)
            axes.set_ylabel(score_name)
        plt.title(name)
        plt.show()