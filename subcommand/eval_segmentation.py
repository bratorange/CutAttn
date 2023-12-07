from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
import os

import numpy as np

from .subcommand import Subcommand, register_subcommand


@register_subcommand
class EvalSeg(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("dataset_root", type=str)

    @staticmethod
    def invoke(experiments, args):
        root = Path(args.dataset_root)
        split_filename = "test.txt"
        use_split = True
        add_circle = True
        resize = True


        import pytorch_lightning as pl
        import torch
        from matplotlib import pyplot as plt
        from torch.utils.data import DataLoader

        from evaluation.model import Model

        from evaluation.split_dataset import SplitDataset

        if use_split:
            split_filepath = root / split_filename
            with open(split_filepath) as f:
                split_data = f.read().strip("\n").split("\n")
            filenames = [x.split(" ")[0] for x in split_data]
        else:
            filenames = [x.stem for x in (root / "images").iterdir()]

        test_dataset =  SplitDataset(root, filenames, add_circle=add_circle, resize=resize)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count())

        trainer = pl.Trainer(
            gpus=1,
            max_epochs=1,
        )

        model = Model.load_from_checkpoint(Path("logs/cholec8K/k=0/checkpoints/epoch=21-valid_per_image_iou=0.8080916404724121-i=0.ckpt"))

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