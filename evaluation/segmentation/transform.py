import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import math

def constant_circle_mask(img, raw_w, raw_h):
    mask_x = 0
    mask_y = 0
    radius = int(raw_w*.45)
    mask = Image.new('L', (raw_w, raw_h), 255)
    draw = ImageDraw.Draw(mask)
    x0 = raw_w * 0.5 - radius + mask_x
    x1 = raw_w * 0.5 + radius + mask_x
    y0 = raw_h * 0.5 - radius + mask_y
    y1 = raw_h * 0.5 + radius + mask_y
    draw.ellipse([x0, y0, x1, y1], fill=0)
    mask = mask.resize(img.size)
    img.paste(0, mask=mask)
    return img