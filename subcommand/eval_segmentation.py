from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
import os

import numpy as np

from evaluation import get_experiment
from evaluation.dataset_creation import create_dataloader
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class EvalSeg(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("--weights_name", default="default")
        subparsers = sc_parser.add_subparsers(dest='mode')

        cut_parser = subparsers.add_parser("cut")
        cut_parser.add_argument('experiment_id', type=int)
        cut_parser.add_argument('--epoch', default='latest')

        test_parser = subparsers.add_parser("test")
        test_parser.add_argument("dataset_root", type=str)

    @staticmethod
    def invoke(experiments, args):
        add_circle = True
        resize = True
        weight_path = Path("logs/weights/") / (args.weights_name + ".ckpt")
        use_split = True
        split_filename = "test.txt"


        import pytorch_lightning as pl
        import torch
        from matplotlib import pyplot as plt
        from evaluation.model import Model


        trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
        )
        test_dataloader = create_dataloader(experiments, args, split_filename, add_circle=add_circle, resize=resize,
                                            shuffle=False, use_split=use_split)
        model = Model.load_from_checkpoint(weight_path)

        # run test dataset
        test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
        pprint(test_metrics)

        batch = next(iter(test_dataloader))

        with torch.no_grad():
            model.eval()
            logits = model(batch["image"])
        pr_masks = logits.sigmoid()

        for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")

            index_pred_mask = torch.argmax(pr_mask, dim=0, keepdim=False)

            plt.subplot(1, 3, 3)
            plt.imshow(index_pred_mask.numpy())
            plt.title("Prediction")
            plt.axis("off")

            plt.show()