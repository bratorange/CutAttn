from PIL import Image, ImageDraw
import random
import math

def circle_mask( img, ox, oy, radius, fill=0 ):
    mask = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(mask)
    x0 = img.size[0]*0.5 - radius + ox
    x1 = img.size[0]*0.5 + radius + ox
    y0 = img.size[1]*0.5 - radius + oy
    y1 = img.size[1]*0.5 + radius + oy
    draw.ellipse([x0,y0,x1,y1], fill=0)
    img.paste((fill,)*3, mask=mask )
    return img

def add_circle_mask(img):
    img_w, img_h = img.width, img.height
    minSize = min((img_w, img_h))
    maxSize = max((img_w, img_h))
    maxRadius = int(math.sqrt((img_w/2)**2 + (img_h/2)**2))
    minRadius = int(0.4*maxSize)
    maskRadius = random.randint( minRadius, maxRadius )
    maskOx = random.randint( int(-img_w*0.1), int(img_w*0.1) )
    maskOy = random.randint( int(-img_h*0.1), int(img_h*0.1) )
    img = circle_mask( img, maskOx, maskOy, maskRadius )
    return img

def constant_circle_mask(img, raw_w, raw_h, fill=0):
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
    img.paste((fill,)*3, mask=mask)
    return img