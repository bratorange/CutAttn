import re
from argparse import ArgumentParser
from typing import Type


class Subcommand:
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        pass

    @staticmethod
    def invoke(experiments, args):
        pass

    @staticmethod
    def get_name() -> str:
        pass


subcommand_types = {}
pattern = re.compile(r'(?<!^)(?=[A-Z])')


# every subcommand registers itself to be used in the launcher
def register_subcommand(sc: Type[Subcommand]):
    # convert class names from camel case to snake case
    name = re.sub(pattern,'_', sc.__name__).lower()
    subcommand_types[name] = sc
