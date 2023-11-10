from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from piq import FID
from torchvision.datasets import VisionDataset
from PIL import Image

######
name = "resnet_atn_08_start_spectral_norm"
epoch = 10
batch_size = 4
######

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

# Define the transforms for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust with the appropriate normalization values
])

# Paths to your directories
prefix = Path('results')/name/f'test_{epoch}'/'images'
fake_images_path = prefix / 'fake_B'
real_images_path = prefix / 'real_B'

# Load images using ImageFolder and create DataLoaders
fake_dataset = DomainDataset(root=fake_images_path, transform=transform)
real_dataset = DomainDataset(root=real_images_path, transform=transform)

fake_dl = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)
real_dl = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

# Initialize FID metric
fid_metric = FID()

print('computing features')
# Compute features using the valid DataLoaders
fake_feats = fid_metric.compute_feats(fake_dl)
real_feats = fid_metric.compute_feats(real_dl)

print('computing feature difference')
# Compute FID
fid_score = fid_metric(real_feats, fake_feats)
print(f"The FID score is: {fid_score.item()}")
