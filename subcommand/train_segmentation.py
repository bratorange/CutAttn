from argparse import ArgumentParser
from pathlib import Path

from evaluation import get_experiment
from evaluation.dataset_creation import create_dataloader
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class TrainSeg(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("--pretrained", type=str, default=None)
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
        if args.pretrained:
            pretrained_path = Path(args.pretrained)
            name = f"{pretrained_path.parts[-4]}_finetuned"
        else:
            name = f"{get_experiment(experiments, args)[2]}_{args.epoch}" if args.mode == "cut" else Path(args.dataset_root).name
        add_circle = True
        resize = True
        ##################################

        # import here to not slow down the launcher
        import pytorch_lightning as pl
        from pytorch_lightning import loggers
        from pytorch_lightning.callbacks import ModelCheckpoint
        from evaluation.model import Model

        train_dataloader, valid_dataloader = create_dataloader(
            experiments, args, split_filename, add_circle=add_circle, resize=resize,
            use_split=True, create_valid=True)
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=100,
            enable_checkpointing=True,
            callbacks=[ModelCheckpoint(
                filename='{epoch}-{per_image_iou}',
                save_top_k=2,
                mode='max',
                monitor='per_image_iou',
                verbose=True,
            )],
            logger=loggers.TensorBoardLogger(
                save_dir="./logs",
                name=name,
            )
        )
        if args.pretrained is None:
            model = Model()
        else:
            model = Model.load_from_checkpoint(args.pretrained)
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
        best_path = trainer.checkpoint_callback.best_model_path
        return best_path
