import re
from argparse import ArgumentParser
from typing import Type


class Subcommand:
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        """
        add parameters for the launcher
        @param sc_parser:
        the ArgumentParser passed by the launcher to populate
        """
        pass

    @staticmethod
    def invoke(experiments, args):
        """
        Invokes the code of this subcommand
        @param experiments:
        The experiments provided by the launcher
        @param args:
        subcommand specific parameters
        """
        pass



subcommand_types = {}
pattern = re.compile(r'(?<!^)(?=[A-Z])')


# every subcommand registers itself to be used in the launcher
def register_subcommand(sc: Type[Subcommand]):
    # convert class names from camel case to snake case
    name = re.sub(pattern,'_', sc.__name__).lower()
    subcommand_types[name] = sc
