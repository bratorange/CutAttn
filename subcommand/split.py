import random
from argparse import ArgumentParser
from pathlib import Path

from .subcommand import Subcommand, register_subcommand


@register_subcommand
class SplitData(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("dataset_root", type=str)

    @staticmethod
    def invoke(experiments, args):
        root = Path(args.dataset_root)

        assert root.is_dir()
        trainval_file = root / 'trainval.txt'
        test_file = root / 'test.txt'

        images_dir = root / "images"

        from evaluation.cholec8k import read_dataset
        images, masks = read_dataset(str(images_dir))

        random.shuffle(images)

        with open(trainval_file, 'w') as f_trainval, open(test_file, 'w') as f_test:
            for i, file_name in enumerate(images):
                if i < 0.6 * len(images):
                    f_trainval.write(file_name[:-4] + '\n')
                else:
                    f_test.write(file_name[:-4] + '\n')
