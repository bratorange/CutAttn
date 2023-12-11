import os
import re
from pathlib import Path

###
target=Path("../cholec8K")
source = Path("/home/schmittju/kp_transformer/cholec8K/")
###

name_pattern = re.compile(r"(frame_[0-9]+).*")
target.mkdir(exist_ok=True)
image_name = target / "images"
image_name.mkdir(exist_ok=True)
mask_name = target / "masks"
mask_name.mkdir(exist_ok=True)

for dir in source.iterdir():
    for video_item in dir.iterdir():
        for frame in video_item.iterdir():
            file_name = frame.name
            name = re.sub(name_pattern, r"\1", file_name)
            name = f"{video_item.name}_{name}.png"

            folder = mask_name if re.match(r".*color_mask.*", file_name) else image_name

            print(f"Linking {frame} to {folder / name}")
            os.symlink(frame, folder / name)
