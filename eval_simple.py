from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from piq import FID, KID
from torchvision.datasets import VisionDataset
from PIL import Image

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

def eval(name, args):
    # Define the transforms for preprocessing the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust with the appropriate normalization values
    ])

    # Paths to your directories
    prefix = Path('results') / name / f'test_{args.epoch}' / 'images'
    fake_images_path = prefix / 'fake_B'
    real_images_path = prefix / 'real_B'

    # Load images using ImageFolder and create DataLoaders
    fake_dataset = DomainDataset(root=fake_images_path, transform=transform)
    real_dataset = DomainDataset(root=real_images_path, transform=transform)

    fake_dl = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)
    real_dl = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize FID metric
    for METRIC in [FID, KID]:
        metric_init = METRIC()
        metric_name = METRIC.__name__
        print(f"Computing {metric_name}...")
        # Compute features using the valid DataLoaders
        fake_feats = metric_init.compute_feats(fake_dl)
        real_feats = metric_init.compute_feats(real_dl)

        # Compute FID
        score = metric_init(real_feats, fake_feats)
        print(f"The {metric_name} score is: {score.item()}")
