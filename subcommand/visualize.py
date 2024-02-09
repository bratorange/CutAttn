from argparse import ArgumentParser

from matplotlib import cm

from evaluation import get_experiment, get_eval_file, get_score_file, get_experiments
from .subcommand import Subcommand, register_subcommand
from pathlib import Path


@register_subcommand
class Vis(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("experiment_id", type=str)
        sc_parser.add_argument("--scores", action="store_true")
        sc_parser.add_argument("--export", action="store_true")

    @staticmethod
    def invoke(experiments, args):

        import numpy as np
        from matplotlib import pyplot as plt
        multiscale = args.scores

        for experiment, epochs, name, args in get_experiments(experiments, args):
            loaded_data = np.load(get_score_file(name) if args.scores else get_eval_file(name))
            metric_names = loaded_data['metric_names']
            scores = loaded_data['scores']
            epochs = loaded_data['epochs']
            if args.export:
                metric_names = np.expand_dims(np.append(["label"], metric_names), axis=1)
                scores = np.char.mod('%f', scores)
                epochs = np.expand_dims(epochs, axis=0)
                scores = np.append(epochs, scores, axis=0)
                table = np.append(metric_names, scores, axis=1)
                path = Path("thesis_data")
                metric_type = 'scores' if args.scores else 'eval'
                path /= metric_type
                path.mkdir(exist_ok=True)
                path  /= f"{name}_{metric_type}.csv"
                table = table.transpose()
                np.savetxt(path, table, delimiter=",", fmt='%s')

            else:
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

                if not args.scores:
                    plt.xticks(np.arange(min(epochs), max(epochs) + 1, 1.0))
                    plt.yticks(np.arange(0, scores.max() + .02, .02))
                plt.legend(metric_names)
                plt.grid()
                plt.title(name)
                plt.show()
