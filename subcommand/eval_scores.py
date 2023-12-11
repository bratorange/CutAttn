import copy
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from evaluation import get_experiment, get_eval_file
from .subcommand import Subcommand, register_subcommand


@register_subcommand
class Score(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=int)
        sc_parser.add_argument('epoch', default='latest')
        sc_parser.add_argument('--batch_size', type=int, default=4)
        sc_parser.add_argument("--dry", action='store_true')

    @staticmethod
    def invoke(experiments, args):

        epoch = args.epoch

        _, _, name = get_experiment(experiments, args)

        print(f"Evaluating {name} at epoch {epoch}")
        if args.dry:
            return

        # import all the heavy stuff only here
        from piq import FID, KID
        from torch.utils.data import DataLoader
        from torchvision.transforms import transforms
        from PIL import Image
        from torchvision.datasets import VisionDataset

        # Define the transforms for preprocessing the images
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust with the appropriate normalization values
        ])

        # Paths to your directories
        prefix = Path('results') / name / f'test_{epoch}' / 'images'
        fake_images_path = prefix / 'fake_B'
        real_images_path = prefix / 'real_B'

        # prevent heavy stuff from being loaded all the time
        class DomainDataset(VisionDataset):
            def __init__(self, root, transform=None, target_transform=None):
                super(DomainDataset, self).__init__(root, transform=transform, target_transform=target_transform)
                root = Path(root)
                self.images = list(root.iterdir())

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                path = str(self.images[index])
                image = Image.open(path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return {'images': image}

        # Load images using ImageFolder and create DataLoaders
        fake_dataset = DomainDataset(root=fake_images_path, transform=transform)
        real_dataset = DomainDataset(root=real_images_path, transform=transform)

        fake_dl = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)
        real_dl = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize FID metric
        metrics = [FID, KID]
        scores = []
        score_names = []
        for METRIC in metrics:
            metric_init = METRIC()
            metric_name = METRIC.__name__
            print(f"Computing {metric_name}...")
            # Compute features using the valid DataLoaders
            fake_feats = metric_init.compute_feats(fake_dl)
            real_feats = metric_init.compute_feats(real_dl)

            # Compute FID
            score = metric_init(real_feats, fake_feats)
            print(f"The {metric_name} score is: {score.item()}")
            scores.append(score)
            score_names.append(metric_name)

        return {"score_names": score_names, "scores": scores}


@register_subcommand
class ScoresAll(Subcommand):
    @staticmethod
    def populate_subparser(sc_parser: ArgumentParser):
        sc_parser.add_argument('experiment_id', type=int)
        sc_parser.add_argument('--batch_size', type=int, default=4)
        sc_parser.add_argument("--dry", action='store_true')

    @staticmethod
    def invoke(experiments, args):
        _, epochs, experiment_name = get_experiment(experiments, args)

        scores = []
        for epoch in epochs:
            downstream_args = copy.deepcopy(args)
            setattr(downstream_args, "epoch", epoch)
            scores.append(Score.invoke(experiments, downstream_args))
        if args.dry:
            return

        # strip metric names out of the return type
        metric_names = np.array(scores[0]["score_names"])

        # scores: turn list(dict(list)) into list(list())
        scores = [result["scores"] for result in scores]
        scores = np.array(scores)
        scores = np.transpose(scores)
        np.savez(get_eval_file(experiment_name), metric_names=metric_names, scores=scores, epochs=epochs)

        print(scores)
