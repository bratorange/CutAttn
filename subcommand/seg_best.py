import argparse
from argparse import ArgumentParser

import numpy as np

from evaluation import get_experiments, get_eval_file
from subcommand import Subcommand, EvalAll, TrainSeg, Eval, register_subcommand


@register_subcommand
class SegBest(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=str)
        sc_parser.add_argument("--weights", default="logs/weights/default.ckpt")

    def invoke(experiments, args):
        # 1. run all eveluations
        print("Phase 1: Mask Preservation Task")
        EvalAll.invoke(experiments, args)

        print("Phase 2: Training new models")
        for experiment, epochs, name, args in get_experiments(experiments, args):
            print(f"{name} is evaluated")
            # 2. get results
            loaded_data = np.load(get_eval_file(name))
            scores = loaded_data['scores']
            epochs = loaded_data['epochs']
            metric_names = loaded_data['metric_names']

            # grab the per_image_iou metrics
            pii_i = list(metric_names).index('per_image_iou')
            scores = scores[pii_i]

            max_epoch_idx = scores.argmax()
            max_epoch = str(epochs[max_epoch_idx])

            # 3. train segmentation from best results
            print(f"Using epoch {max_epoch} for pretraining")
            train_args = argparse.Namespace(mode='cut', experiment_id=args.experiment_id, epoch=max_epoch, pretrained=None)
            best_path = TrainSeg.invoke(experiments, train_args)
            print(f"Pretraining done for {name} epoch {max_epoch}. Finetuning...")
            train_args = argparse.Namespace(mode='test', dataset_root='cholec8K', pretrained=best_path)
            best_path = TrainSeg.invoke(experiments, train_args)
            print(f"Finetuning done for {name} epoch {max_epoch}. Evaluating {best_path}")

            # 4. eval trained models
            eval_args = argparse.Namespace(weights=best_path, mode='test', dataset_root='cholec8K', save=True)
            test_metrics = Eval.invoke(experiments, eval_args, show=False)
            print(f"Received test_metrics for {name} epoch {max_epoch}:")
            print(test_metrics)