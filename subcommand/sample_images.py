import argparse

import numpy as np

from evaluation import get_experiment
from .subcommand import Subcommand, register_subcommand
from argparse import ArgumentParser


@register_subcommand
class SampleImgs(Subcommand):

    # let a command look like
    # sample_imgs -m 52 18 -m 41 5 --pick img42.png
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('-m', nargs=2, action='append', dest='models', metavar=('experiment_id', 'epoch'),
                               help='Choose an experiment and epoch')

        # Adding the '--pick' argument to specify the image file
        sc_parser.add_argument('--pick', metavar='image_files', help='Specify the image file')

    @staticmethod
    def invoke(experiments, args):
        image_files = args.pick.split(",")
        print(f"Sampling {image_files} from:")
        models = np.array(args.models)
        ids = models[:, 0]
        epochs = models[:, 1]
        for id, epoch in zip(ids, epochs):
            _, _, name = get_experiment(experiments, argparse.Namespace(experiment_id=int(id)))
            print(f"{name} at epoch {epoch}")
