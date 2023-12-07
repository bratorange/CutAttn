from pathlib import Path

import torch
from collections import namedtuple
import os
from typing import Tuple, List, Dict, Optional, Union

color_code = namedtuple('colorcode', ['name', 'color'])

label_codes = [color_code(name='bg', color=(127,127,127)),
               color_code(name='abwall', color=(210,140,140)),
               color_code(name='liver', color=(255,114,114)),
               color_code(name='fat', color=(186,183,75)),
               color_code(name='grasper', color=(170,255,0)),
               color_code(name='hook', color=(169,255,184)),
               color_code(name='ligament', color=(111,74,0)),
               color_code(name='gall', color=(255,160,165))]


def label_to_channel(col_mask, labelColors=label_codes):
    """
    Function to modify the labels in Cholec8K dataset
    - There are 8 classes in total. The hook & grasper are fused 
        into a single class in this work

    Input:
        col_mask: torch Tensor (Longtensor)
            The PIL image converted to torch tensor

    Return:
        lbl: torch tensor of labels with values btw 0 and 6 
    """
    lbl = torch.zeros(col_mask.shape[1:], dtype=torch.int64)
    lbl.fill_(255)
    for i in range( 0, len(labelColors) ):
        lc = labelColors[i]
        if i>=5:
            i-=1
        mask = (col_mask[0,:,:] == lc.color[0])*(col_mask[1,:,:] == lc.color[1])*(col_mask[2,:,:] == lc.color[2])
        #if lc.name == "hook":
        #    lbl[mask] = 4
        lbl[mask] = i
    
    lbl[lbl==255] = 0
    return lbl

def read_dataset(main_path: str) -> list:
    
    """
    Custom function to read the images and masks of cholec8K dataset
    stored in different different folders

    Input:
        main_path: the path to the dataset
            default at '/mnt/ceph/tco/TCO-Students/Projects/KP_transformer/cholec8K/'
    
    Return:
        sorted image and mask files
    """
    mask_files = []
    img_files = []
    c=0
    
    for root, dirs, files in os.walk(main_path):
        for f in files:
            if f[-4:] == ".png" and not ("mask" in f):
                img_files.append(os.path.join(root, f))
            else:
                mask_files.append(os.path.join(root, f))
    
    img_files.sort()
    mask_files.sort()
    return img_files, mask_files
