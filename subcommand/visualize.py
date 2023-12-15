from argparse import ArgumentParser

from matplotlib import cm

from evaluation import get_experiment, get_eval_file, get_score_file
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class VisScores(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("experiment_id", type=int)

    @staticmethod
    def invoke(experiments, args):

        import numpy as np
        from matplotlib import pyplot as plt
        multiscale = False

        experiment, epochs, name = get_experiment(experiments, args)

        loaded_data = np.load(get_eval_file(name))
        metric_names = loaded_data['metric_names']
        scores = loaded_data['scores']
        epochs = loaded_data['epochs']

        colors = cm.rainbow(np.linspace(0, 1, len(metric_names)))

        for i, (score_name, score, color) in enumerate(zip(metric_names, scores, colors)):
            if multiscale:
                if i == 0:
                    fig, axes = plt.subplots()
                    axes.set_xlabel("epoch")
                else:
                    axes = axes.twinx()
                axes.plot(epochs, score, marker="v", color=color)
                axes.set_ylabel(score_name)
            else:
                plt.plot(epochs, score, marker="v", color=color)

        plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
        plt.yticks(np.arange(0, scores.max() + .02, .02))
        plt.legend(metric_names)
        plt.grid()
        plt.title(name)
        plt.show()
