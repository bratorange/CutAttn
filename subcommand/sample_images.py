from .subcommand import Subcommand, register_subcommand
from argparse import ArgumentParser

@register_subcommand
class SampleImgs(Subcommand):

    # let a command look like
    # sample_imgs -m 52 18 -m 41 5 --pick img42.png
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):

    @staticmethod
    def invoke(experiments, args):
