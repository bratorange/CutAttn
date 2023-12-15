import os
from collections import namedtuple

import torch

color_code = namedtuple('colorcode', ['name', 'color'])

label_names = ['background', 'abwall', 'liver', 'fat', 'tool', 'ligament', 'gallbladder']
n_classes = len(label_names)

label_codes = {
    'cholec8K':
    [
        color_code(name='bg', color=(127,127,127)),
        color_code(name='abwall', color=(210,140,140)),
        color_code(name='liver', color=(255,114,114)),
        color_code(name='fat', color=(186,183,75)),
        color_code(name='grasper', color=(170,255,0)),
        color_code(name='hook', color=(169,255,184)),
        color_code(name='ligament', color=(111,74,0)),
        color_code(name='gall', color=(255,160,165))
    ],
    'li2it':
    [
        color_code( name="Void", color=0 ),
        color_code(name="Diaphragm", color=77),
        color_code( name="Liver", color=26 ),
        color_code( name="Fat", color=102 ),
        color_code(name="ToolTip", color=153),
        color_code(name="ToolShaft", color=179),
        color_code( name="Ligament", color=128 ),
        color_code( name="Gallbladder", color=51 ),
    ],

}


def label_to_channel(col_mask, labelColors):
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
    lbl = torch.zeros((1,) + col_mask.shape[1:], dtype=torch.long)
    for i in range( 0, len(labelColors) ):
        lc = labelColors[i]
        if i>=5:
            i-=1
        if type(lc.color) is int:
            mask = (col_mask[0, :, :] == lc.color)
        else:
            mask = (col_mask[0,:,:] == lc.color[0])*(col_mask[1,:,:] == lc.color[1])*(col_mask[2,:,:] == lc.color[2])
        #if lc.name == "hook":
        #    lbl[mask] = 4
        lbl[0][mask] = i

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
