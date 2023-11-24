import os
from pathlib import Path

###
target=Path("imagenet")
source = Path("/mnt/ceph/tco/TCO-All/Projects/ImageNet-1k/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train")
###

target.mkdir(exist_ok=True)
trainB = target / "trainB"
trainB.mkdir(exist_ok=True)

for dir in source.iterdir():
    for item in dir.iterdir():
        print(item.name)
        os.symlink(item, trainB/item.name)