import os
import glob
from tqdm import tqdm

from PIL import Image

target_res=128

output_path='./processed_images'

os.makedirs(output_path, exist_ok=True)

for infile in tqdm(glob.glob('dev_images/*')):
    im = Image.open(infile)

    width, height = im.size
    min_side = min(width, height)
    if min_side < target_res:
        continue

    _, fname = os.path.split(infile)
    fname, _ = os.path.splitext(fname)

    im = im.resize((target_res, target_res), Image.BICUBIC)
    im.save(f'{output_path}/{fname}.jpg', 'JPEG')
