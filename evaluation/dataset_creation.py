import os
from pathlib import Path

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from evaluation import get_experiment
from evaluation.cholec8k import label_codes
from evaluation.split_dataset import SplitDataset


def create_dataloader(experiments, args, split_filename, add_circle=False, resize=False, shuffle=False, use_split=True,
                      k_fold=0, create_valid=False, ):
    use_cut_output = args.mode == "cut"

    if use_cut_output:
        _, _, experiment_name = get_experiment(experiments, args)
        root = Path("results/") / experiment_name / ("test_" + args.epoch) / "images"
    else:
        root = Path(args.dataset_root)
    use_split = False if use_cut_output else use_split

    image_folder = root / "fake_B" if use_cut_output else root / "images"
    mask_folder = Path("dataset/testA_label") if use_cut_output else root / "masks"
    colorcodes = label_codes['li2it'] if use_cut_output else label_codes['cholec8K']

    if use_split:
        split_filepath = root / split_filename
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
    else:
        filenames = [x.name for x in image_folder.iterdir()]

    if k_fold == 0:
        if create_valid:
            train = [x for i, x in enumerate(filenames) if i % 10 != 0]
            valid = [x for i, x in enumerate(filenames) if i % 10 == 0]
            return (
                DataLoader(
                    SplitDataset(train, image_folder, mask_folder, colorcodes, add_circle=add_circle, resize=resize),
                    batch_size=8, shuffle=False, num_workers=0),
                DataLoader(
                    SplitDataset(valid, image_folder, mask_folder, colorcodes, add_circle=add_circle, resize=resize),
                    batch_size=8, shuffle=False, num_workers=0),
            )
        else:
            dataset = SplitDataset(filenames, image_folder, mask_folder, colorcodes, add_circle=add_circle,
                                   resize=resize)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=shuffle, num_workers=0)
            return dataloader

    else:
        # do the kfold splits
        split_filenames = [
            [
                [split_data[y] for y in train],
                [split_data[y] for y in valid],
            ]
            for train, valid in KFold(n_splits=k_fold, shuffle=True, random_state=False).split(filenames)]

        dataloaders = [(
            DataLoader(SplitDataset(train, image_folder, mask_folder, colorcodes, add_circle=add_circle, resize=resize),
                       batch_size=16, shuffle=shuffle,),
            DataLoader(SplitDataset(valid, image_folder, mask_folder, colorcodes, add_circle=add_circle, resize=resize),
                       batch_size=16, shuffle=shuffle,),
        )
            for train, valid in split_filenames
        ]
        return dataloaders
