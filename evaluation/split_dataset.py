import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.transform import constant_circle_mask
from evaluation.cholec8k import label_to_channel


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, root, filenames, add_circle=False, transform=None, resize=False):
        self.resize = resize
        self.add_circle = add_circle
        self.transform = transform
        self.root = root
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        image_path = filename + '.png'
        mask_path = filename + '_color_mask.png'
        transform = transforms.ToTensor()

        mask = Image.open(mask_path).convert('RGB')
        raw_w, raw_h = mask.size
        if self.resize:
            mask = mask.resize((256, 256))
        if self.add_circle:
            if self.resize:
                mask = constant_circle_mask(mask, raw_w, raw_h)
            else:
                mask = constant_circle_mask(mask, mask.width, mask.height)
        mask = np.transpose(np.array(mask), (2, 0, 1))
        mask = label_to_channel(mask)
        mask = mask.unsqueeze(0)

        image = Image.open(image_path).convert('RGB')
        if self.resize:
            image = image.resize((256, 256))
        if self.add_circle:
            if self.resize:
                image = constant_circle_mask(image, raw_w, raw_h)
            else:
                image = constant_circle_mask(image, image.width, image.height)
        image = image.convert("RGB")
        image = transform(image)

        sample = dict(image=image, mask=mask, filename=filename)
        return sample
