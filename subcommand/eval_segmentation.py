import copy
import json
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import numpy as np

from evaluation import get_experiments, get_eval_file, get_experiment
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class EvalAll(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=str)
        sc_parser.add_argument("--weights", default="default")

    def invoke(experiments, args):
        for experiment, epochs, name, args in get_experiments(experiments, args):
            iou = []
            for epoch in epochs:
                print(f"Evaluating iou scores for epoch {epoch} from {name}")
                args = copy.deepcopy(args)
                args.epoch = str(epoch)
                args.mode = 'cut'
                args.save = False
                iou.append(Eval.invoke(experiments, args, show=False))

            scores = [[*x.values()] for x in iou]
            scores = np.array(scores).transpose()

            metric_names = [*iou[0].keys()]
            np.savez(get_eval_file(name), metric_names=metric_names, scores=scores, epochs=epochs)

@register_subcommand
class Eval(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument("--weights", default="default")
        subparsers = sc_parser.add_subparsers(dest='mode')
        sc_parser.add_argument("--save", action="store_true")

        cut_parser = subparsers.add_parser("cut")
        cut_parser.add_argument('experiment_id', type=int)
        cut_parser.add_argument('--epoch', default='latest')

        test_parser = subparsers.add_parser("test")
        test_parser.add_argument("dataset_root", type=str)

    @staticmethod
    def invoke(experiments, args, show=True):
        add_circle = True
        resize = True
        weight_path = Path("logs/weights/") / (args.weights + ".ckpt")
        use_split = True
        split_filename = "test.txt"
        if args.mode == "cut":
            _, _, experiment_name = get_experiment(experiments, args)
            name = f"{experiment_name}_{args.epoch}"
        else:
            name = args.dataset_root


        import pytorch_lightning as pl
        import torch
        from matplotlib import pyplot as plt
        from evaluation.model import Model
        from evaluation.dataset_creation import create_dataloader


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

        test_metrics = test_metrics[0]
        if args.save:
            file = Path("thesis_data") / f"{args.weights}_{name}.json"
            with open(file, "w") as fd:
                json.dump(test_metrics, fd)

        if show:
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


        return test_metrics
