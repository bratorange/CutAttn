import os
from datetime import datetime
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from split_dataset import SimpleSplitDataset
from model import Model

##########  Variables   ##########
split_filename = "trainval.txt"
root = "./train_data"
n_splits = 3
name = "kp_transformer"
mask_value = 255
add_circle = True
resize = False
##################################

# load image names
split_filepath = os.path.join(root, split_filename)
with open(split_filepath) as f:
    split_data = f.read().strip("\n").split("\n")
    filenames = [x.split(" ")[0] for x in split_data]

# do the kfold splits
test = list(KFold(n_splits=n_splits, shuffle=True, random_state=False).split(filenames))
split_filenames = [
    [
        [split_data[y] for y in train],
        [split_data[y] for y in valid],
    ]
    for train, valid in KFold(n_splits=n_splits, shuffle=True, random_state=False).split(filenames)]

datasets = [(
    DataLoader(SimpleSplitDataset(root, train, mask_value=mask_value, add_circle=add_circle, resize=resize), batch_size=16, shuffle=True, num_workers=os.cpu_count()),
    DataLoader(SimpleSplitDataset(root, valid, mask_value=mask_value, add_circle=add_circle, resize=resize), batch_size=16, shuffle=True, num_workers=os.cpu_count()),
)
    for train, valid in split_filenames
]

for k, (train_dataloader, valid_dataloader) in enumerate(datasets):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        enable_checkpointing=True,
        callbacks=[ModelCheckpoint(
            filename='{epoch}-{valid_per_image_iou}-{i}',
            save_top_k=1,
            mode='max',
            monitor='valid_per_image_iou',
            verbose=True,
        )],
        logger = loggers.TensorBoardLogger(
            version=f"k={k}",
            save_dir="./logs",
            name = name,
        )
    )

    model = Model()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
