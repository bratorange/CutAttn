import copy
import os
from argparse import ArgumentParser

from evaluation import get_epochs
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class TestAll(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=str)
        sc_parser.add_argument("--dry", action='store_true')
        sc_parser.add_argument('--batch_size', type=int, default=4)
        sc_parser.add_argument("--num_test", type=int, default=50)

    @staticmethod
    def invoke(experiments, args):
        experiment, epochs, name = get_epochs(experiments, args)
        for epoch in epochs:
            command = "python test.py " + str(
                copy.deepcopy(experiment).set(epoch=epoch, num_test=args.num_test).remove('continue_train', "lr",
                                                                                          "pretrained_name", ))
            print(f"Infering epoch {epoch}...")
            print(command)
            if not args.dry:
                if os.system(command) != 0:
                    exit()


@register_subcommand
class Test(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=str)
        sc_parser.add_argument("--dry", action='store_true')
        sc_parser.add_argument('--batch_size', type=int, default=4)
        sc_parser.add_argument("--num_test", type=int, default=50)
        sc_parser.add_argument('epoch', default='latest')

    @staticmethod
    def invoke(experiments, args):
        command = "python test.py " + str(
            experiments[args.experiment_id].set(epoch=args.epoch, num_test=args.num_test).remove('continue_train', "lr",
                                                                                                 "pretrained_name", ))
        print(command)
        if not args.dry:
            os.system(command)
