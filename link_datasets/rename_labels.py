import os
import re
from pathlib import Path

folder = Path("/home/schmittju/kp_transformer/testA_label")

for file in folder.iterdir():
    name_new = re.sub(r"lbl", r"img", file.name)
    print(f"rename {file.name} to {name_new}")
    os.rename(file, folder / name_new)