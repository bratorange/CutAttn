from PIL import Image, ImageDraw
import random
import math

def circle_mask( img, ox, oy, radius ):
    mask = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(mask)
    x0 = img.size[0]*0.5 - radius + ox
    x1 = img.size[0]*0.5 + radius + ox
    y0 = img.size[1]*0.5 - radius + oy
    y1 = img.size[1]*0.5 + radius + oy
    draw.ellipse([x0,y0,x1,y1], fill=0)
    img.paste( (0,0,0), mask=mask )
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

def constant_circle_mask(img):
    img_w, img_h = img.width, img.height
    mask_x = 0
    mask_y = 0
    radius = int(img_w*.45)
    return circle_mask(img, mask_x, mask_y, radius)