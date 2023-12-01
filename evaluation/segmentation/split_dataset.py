import os
import re

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from transform import constant_circle_mask


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, root, filenames, mask_value, add_circle=False, transform=None, resize=False):

        self.resize = resize
        self.add_circle = add_circle
        self.transform = transform

        self.root = root
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "labels")

        self.mask_value = mask_value
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]

        image_path = os.path.join(self.images_directory, filename + '.png')
        mask_path = os.path.join(self.masks_directory, re.sub("img", "lbl", filename) + '.png')

        trimap = Image.open(mask_path)
        raw_w, raw_h = trimap.size
        if self.resize:
            trimap = trimap.resize((256, 256))
        if self.add_circle:
            if self.resize:
                trimap = constant_circle_mask(trimap, raw_w, raw_h)
            else:
                trimap = constant_circle_mask(trimap, trimap.width, trimap.height)
        trimap = trimap.convert("L")
        trimap = np.array(trimap)
        mask = self._preprocess_mask(trimap)

        image = Image.open(image_path)
        if self.resize:
            image = image.resize((256, 256))
        if self.add_circle:
            if self.resize:
                image = constant_circle_mask(image, raw_w, raw_h)
            else:
                image = constant_circle_mask(image, image.width, image.height)
        image = image.convert("RGB")
        image = np.array(image)


        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def _preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask != self.mask_value] = 0.0
        mask[mask == self.mask_value] = 1.0
        return mask


class SimpleSplitDataset(SplitDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.LINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample
