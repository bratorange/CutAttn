import os

from torch.utils.data import DataLoader

from evaluation.trainval_dataset import SimpleTrainValDataset


def get_dataloaders():
    root = "."

    train_dataset = SimpleTrainValDataset(root, "train")
    valid_dataset = SimpleTrainValDataset(root, "valid")
    test_dataset =  SimpleTrainValDataset(root, "test")

    # It is a good practice to check datasets don`t intersects with each other
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    return train_dataloader, valid_dataloader, test_dataloader