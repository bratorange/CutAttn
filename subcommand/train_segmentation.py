from argparse import ArgumentParser
from pathlib import Path

from .subcommand import Subcommand, register_subcommand


@register_subcommand
class TrainSeg(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("dataset_root", type=str)
        sc_parser.add_argument("--n_splits", type=int, default=3)

    @staticmethod
    def invoke(experiments, args):

        ##########  Variables   ##########
        split_filename = "trainval.txt"
        root = Path(args.dataset_root)
        n_splits = args.n_splits
        name = root.name
        add_circle = True
        resize = True
        ##################################

        # import here to not slow down the launcher
        import os
        import pytorch_lightning as pl
        from pytorch_lightning import loggers
        from pytorch_lightning.callbacks import ModelCheckpoint
        from sklearn.model_selection import KFold
        from torch.utils.data import DataLoader
        from evaluation.model import Model
        from evaluation.split_dataset import SplitDataset

        # load image names
        split_filepath = root / split_filename
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
            filenames = [x.split(" ")[0] for x in split_data]

        # do the kfold splits
        split_filenames = [
            [
                [split_data[y] for y in train],
                [split_data[y] for y in valid],
            ]
            for train, valid in KFold(n_splits=n_splits, shuffle=True, random_state=False).split(filenames)]

        datasets = [(
            DataLoader(SplitDataset(root, train, add_circle=add_circle, resize=resize), batch_size=16, shuffle=True, num_workers=os.cpu_count()),
            DataLoader(SplitDataset(root, valid, add_circle=add_circle, resize=resize), batch_size=16, shuffle=True, num_workers=os.cpu_count()),
        )
            for train, valid in split_filenames
        ]

        # train one model for every split
        for k, (train_dataloader, valid_dataloader) in enumerate(datasets):
            print(f"\nTraining split {k+1} of {len(datasets)}")
            trainer = pl.Trainer(
                gpus=1,
                max_epochs=1,
                enable_checkpointing=True,
                callbacks=[ModelCheckpoint(
                    filename='{epoch}-{valid_per_image_iou}-{i}',
                    save_top_k=1,
                    mode='max',
                    monitor='valid_per_image_iou',
                    verbose=True,
                )],
                logger = loggers.TensorBoardLogger(
                    version=f"k={k}",
                    save_dir="./logs",
                    name = name,
                )
            )

            model = Model()
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloader,
            )
