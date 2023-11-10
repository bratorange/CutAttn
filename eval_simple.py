from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from piq import FID


######
name = ""
epoch = 5
batch_size = 4
######

# Define the transforms for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust with the appropriate normalization values
])

# Paths to your directories
prefix = Path('results')/name/f'test_(epoch)'
fake_images_path = prefix / 'fake_B'
real_images_path = prefix / 'real_B'

# Load images using ImageFolder and create DataLoaders
fake_dataset = ImageFolder(root=fake_images_path, transform=transform)
real_dataset = ImageFolder(root=real_images_path, transform=transform)

fake_dl = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)
real_dl = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

# Initialize FID metric
fid_metric = FID()

# Compute features using the valid DataLoaders
fake_feats = fid_metric.compute_feats(fake_dl)
real_feats = fid_metric.compute_feats(real_dl)

# Compute FID
fid_score = fid_metric(fake_feats, real_feats)
print(f"The FID score is: {fid_score.item()}")
