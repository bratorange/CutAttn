import os
from pathlib import Path

###
target=Path("cholec8K")
source = Path("/home/schmittju/kp_transformer/cholec8K/")
###

target.mkdir(exist_ok=True)
trainB = target
trainB.mkdir(exist_ok=True)

for dir in source.iterdir():
    for video_item in dir.iterdir():
        for frame in video_item.iterdir():
            print(frame.name)
            os.symlink(frame, trainB / f"{video_item.name}_{frame.name}")