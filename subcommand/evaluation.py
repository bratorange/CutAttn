import os
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from evaluation.model import Model

from evaluation.split_dataset import SplitDataset

root = Path("./generated/small_epoch")
use_split = False
mask_value = 26
add_circle = True
resize = True



if use_split:
    split_filename = "test.txt"
    split_filepath = os.path.join(root, split_filename)
    with open(split_filepath) as f:
        split_data = f.read().strip("\n").split("\n")
    filenames = [x.split(" ")[0] for x in split_data]
else:
    filenames = [x.stem for x in (root / "images").iterdir()]

test_dataset =  SplitDataset(root, filenames, mask_value=mask_value, add_circle=add_circle, resize=resize)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count())

trainer = pl.Trainer(
    gpus=1,
    max_epochs=1,
)

model = Model.load_from_checkpoint("logs/kp_transformer/k=1/checkpoints/epoch=94-valid_per_image_iou=0.8190122246742249-i=0.ckpt")

# run test dataset
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
pprint(test_metrics)

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
    plt.imshow(gt_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.numpy().squeeze()) # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")

    plt.show()
