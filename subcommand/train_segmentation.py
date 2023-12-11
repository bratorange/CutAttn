from argparse import ArgumentParser
from pathlib import Path

from evaluation import get_experiment
from evaluation.dataset_creation import create_dataloader
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class TrainSeg(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("--n_splits", type=int, default=3)
        subparsers = sc_parser.add_subparsers(dest='mode')

        cut_parser = subparsers.add_parser("cut")
        cut_parser.add_argument('experiment_id', type=int)
        cut_parser.add_argument('--epoch', default='latest')

        test_parser = subparsers.add_parser("test")
        test_parser.add_argument("dataset_root", type=str)

    @staticmethod
    def invoke(experiments, args):

        ##########  Variables   ##########
        split_filename = "trainval.txt"
        n_splits = args.n_splits
        name = get_experiment(experiments, args)[2] if args.mode == "cut" else Path(args.dataset_root).name
        add_circle = True
        resize = True
        shuffle = True
        ##################################

        # import here to not slow down the launcher
        import pytorch_lightning as pl
        from pytorch_lightning import loggers
        from pytorch_lightning.callbacks import ModelCheckpoint
        from evaluation.model import Model

        datasets = create_dataloader(experiments, args, split_filename, add_circle=add_circle, resize=resize, shuffle=shuffle, use_split=True, k_fold=n_splits)
        # train one model for every split
        for k, (train_dataloader, valid_dataloader) in enumerate(datasets):
            print(f"\nTraining split {k+1} of {len(datasets)}")
            trainer = pl.Trainer(
                gpus=1,
                max_epochs=100,
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
