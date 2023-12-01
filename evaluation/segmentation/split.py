import os
import re
import shutil
import random
from pathlib import Path


folders_path = [str(x) for x in Path("/mnt/ceph/tco/TCO-Students/Projects/KP_transformer/liver/").iterdir()]

root = Path('./train_data')
trainval_file = root / 'trainval.txt'
test_file = root / 'test.txt'

os.makedirs(root/'images', exist_ok=True)
os.makedirs(root/'labels', exist_ok=True)

images = []
trimaps = []
for folder_path in folders_path:
    for file_name in os.listdir(folder_path):
        if file_name.startswith('image'):
            shutil.copy(os.path.join(folder_path, file_name), root / 'images' / (folder_path[-2:] + '_' + re.sub("image", "img", file_name)))
            images.append(folder_path[-2:] + '_' + re.sub("image", "img", file_name))
        elif file_name.startswith('mask'):
            shutil.copy(os.path.join(folder_path, file_name), root / 'labels'/ ( folder_path[-2:] + '_' + re.sub("mask", "lbl", file_name)))
            trimaps.append(folder_path[-2:] + '_' + re.sub("image", "lbl", file_name))
    
random.shuffle(images)
# random.shuffle(trimaps)

with open(trainval_file, 'w') as f_trainval, open(test_file, 'w') as f_test:
    for i, file_name in enumerate(images):
        if i < 0.6 * len(images):
            f_trainval.write(file_name[:-4] + '\n')
        else:
            f_test.write(file_name[:-4] + '\n')


