import copy
from argparse import ArgumentParser

from . import TestAll, ScoresAll, Vis
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class TestScoreAll(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=str)
        sc_parser.add_argument("--dry", action='store_true')
        sc_parser.add_argument('--batch_size', type=int, default=4)
        sc_parser.add_argument("--num_test", type=int, default=50)

    @staticmethod
    def invoke(experiments, args):
        experiment_ids = [int(i) for i in args.experiment_id.split(',')]
        for experiment_id in experiment_ids:
            args = copy.deepcopy(args)
            args.experiment_id = int(experiment_id)
            TestAll.invoke(experiments, args)
            ScoresAll.invoke(experiments, args)
            Vis.invoke(experiments, args)
