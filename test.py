from tkinter import image_names
import torch 
from torch import nn
import cv2 
import numpy as np
from skimage import segmentation
import rasterio
import glob
import os
from tqdm.contrib.telegram import tqdm, trange


torch.cuda.manual_seed_all(1943)
np.random.seed(1943)

train_epoch = 2 ** 6
mod_dim1 = 64 
mod_dim2 = 32
gpu_id = 2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
min_label_num = 4
max_label_num = 256  


model = torch.jit.load('path_to_model')
model.eval()
# CHANGE THIS
pngs_path = 'path_to_pngs_dir'
# CHANGE THIS
raster_path = 'path_to_raster'
# CHANGE THIS
result_path = 'path_to_save_dir'
os.makedirs(result_path, exist_ok=True)

im_names = os.listdir(raster_path)
ready_files = os.listdir(result_path)

for name in pbar:
    print(name)
    pbar.set_postfix_str(name)

    if name+'.tif' in ready_files:
        continue

    tif_img_path = raster_path+name[:-4]+'.tif'
    print(tif_img_path)
    eve = 0
    meta = 0
    with rasterio.open(tif_img_path, 'r') as f:
        eve = f.read()
        meta = f.meta

    result_image = np.zeros((1, eve.shape[1], eve.shape[2]))

    input_image_path = pngs_path+name[:-8]+'.png'
    eve = cv2.imread(input_image_path)

    tile_size = 2048
 
    for i in range(0, eve.shape[0], tile_size):
        for j in range(0, eve.shape[1], tile_size):
            image = eve[i : i + tile_size, j:j+tile_size, :]
            orig_shape0 = image.shape[0]
            orig_shape1 = image.shape[1]

            device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

            tensor = image.transpose((2, 0, 1))
            tensor = tensor.astype(np.float32) / 255.0
            tensor = tensor[np.newaxis, :, :, :]
            tensor = torch.from_numpy(tensor).to(device)

            output = model(tensor)[0]

            output = output.permute(1, 2, 0).view(-1, mod_dim2)
            target = torch.argmax(output, 1)
            im_target = target.data.cpu().numpy()

            '''segmentation ML'''
            seg_map = segmentation.felzenszwalb(image, scale=228, sigma=0.6, min_size=1337)
            seg_map = seg_map.flatten()
            seg_lab = [np.where(seg_map == u_label)[0]
                        for u_label in np.unique(seg_map)]

            '''refine'''
            for inds in seg_lab:
                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                im_target[inds] = u_labels[np.argmax(hist)]

            print(im_target.shape)
            im_target = im_target.reshape((orig_shape0, orig_shape1))


            a, b = 2048, 2048
            if result_image[0, i : i + tile_size, j:j+tile_size].shape != im_target.shape and im_target is not None:
                a, b = result_image[0, i : i + tile_size, j:j+tile_size].shape
                if a > im_target.shape[0]:
                    a = im_target.shape[0]
                if b > im_target.shape[1]:
                    b = im_target.shape[1]
                im_target = im_target[0:a, 0:b]
            
            
            
            result_image[0, i : i + a, j:j+b] = im_target[:a, :b]

    meta.update({"driver": "GTiff",
                'dtype':rasterio.uint8,
                    "nodata": 255,
                    'count':1})


    with rasterio.open(result_path+name+'.tif', 'w', **meta) as dest:
        dest.write(result_image.astype(rasterio.uint8))

