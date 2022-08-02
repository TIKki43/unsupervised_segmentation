import rasterio
import glob
import cv2
import numpy as np
import os

# CHANGE THIS
pngs_path = 'path_to_dir_with_png'
images_paths = glob.glob(pngs_path+'*.png')

for im_path in images_paths:
    output_path = im_path[:-4]+'/'
    os.makedirs(output_path, exist_ok=True)

    tile_size = 2048

    image = cv2.imread(im_path)
    print(im_path)
    for i in range(0, image.shape[0], tile_size):
        for j in range(0, image.shape[1], tile_size):
            tile = image[i : i + tile_size, j:j+tile_size, :]
            # print('%s%d_%d.png'%(output_path,i,j))
            cv2.imwrite('%s%d_%d.png'%(output_path,i,j), tile)
