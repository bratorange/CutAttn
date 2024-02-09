import argparse
import random
import shutil
from pathlib import Path

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
        sc_parser.add_argument('-n', dest="n_images", type=int, default=5)

    @staticmethod
    def invoke(experiments, args):
        folder = Path("thesis_data/images")

        folder.mkdir(exist_ok=True)

        if args.pick:
            image_files = args.pick.split(",")
        else:
            all_images = list(Path("dataset/testA_label").iterdir())
            random.shuffle(all_images)
            image_files = all_images[:args.n_images]
        print(f"Sampling images from:")
        models = np.array(args.models)
        ids = models[:, 0]
        epochs = models[:, 1]
        for id, epoch in zip(ids, epochs):
            id = int(id)
            epoch = int(epoch)
            _, _, name = get_experiment(experiments, argparse.Namespace(experiment_id=id))
            print(f"{name} at epoch {epoch}")

            target = folder / f"{name}_{epoch}"
            target.mkdir(exist_ok=True)
            for img_name in image_files:
                img_name = img_name.name
                mask_path = Path("dataset/testA_label") / img_name
                realA_path = Path(f"results/{name}/test_{epoch}/images/real_A") / img_name
                realB_path = Path(f"results/{name}/test_{epoch}/images/real_B") / img_name
                fakeB_path = Path(f"results/{name}/test_{epoch}/images/fake_B") / img_name

                target_path = target / img_name
                target_path.mkdir(exist_ok=True)

                print("\t", mask_path)
                print("\t", realA_path)
                print("\t", realB_path)
                print("\t", fakeB_path)
                print()

                shutil.copy(mask_path, target_path/"mask.png")
                shutil.copy(realA_path, target_path/"real_A.png")
                shutil.copy(realB_path, target_path/"real_B.png")
                shutil.copy(fakeB_path, target_path/"fake_B.png")
