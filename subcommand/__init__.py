# import all modules with subcommands to be registered
from .subcommand import *
from .split import *
from .train_segmentation import *
from .visualize import *

if __name__ == "__main__":
    print(subcommand_types)