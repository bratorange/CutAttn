from .subcommand import *
from .split import *
from .train_segmentation import *
from .visualize import *

if __name__ == "__main__":
    print(subcommand_types)
    obj = subcommand_types[0].invoke()
    print(obj)