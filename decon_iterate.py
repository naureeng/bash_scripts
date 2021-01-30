# -*- coding: utf-8 -*-

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from skimage import io
import tqdm
import skimage
import glob
import numpy as np
import os

wavelengths = [561, 647]
path = '/nfs/winstor/isogai/stella/processed/180712Merfish_3D/BitComposite/Chromatic/Hyb1/*.tif'
psfs = [io.imread('PSF%s_38z.tif' % w) for w in wavelengths]
algo = fd_restoration.RichardsonLucyDeconvolver(3, pad_mode='none').initialize()
images = {}


# Python3 code to convert tuple  
# into string 
def convertTuple(tup): 
    str =  ''.join(tup) 
    return str

# Using tqdm to report progress
for filename in tqdm.tqdm(glob.glob(path)):
    # Assume each image would load as [z, x, y, channels] and that there are two
    # channels (first at 561 nm, second at 657 nm)
    img = skimage.io.imread(filename)
    [z, channels, x, y] = img.shape
    # Move channels axis (1) to last axis (-1) to give result shape [z, x, y, channels]
    img_final = np.moveaxis(img, 1, -1) 
    # Alternatively this will accomplish the same thing in a more general way: 
    # img_final = np.transpose(img, (0, 2, 3, 1))
    assert img_final.ndim == 4
    assert img_final.shape[-1] == 2 # Make sure there are 2 channels
    
    res = [
            algo.run(fd_data.Acquisition(img_final[..., i], psfs[i]), niter=25).data
            for i in range(img_final.shape[-1])
            ]
    
    # Create new axis at the end so that result is also [z,y,x, channels],
    # not [channels, z,y,x]
    res = np.stack(res, -1)
    
    filename_deconv = os.path.splitext(filename)
    final_name = filename_deconv[0] + "_D{}.tif",
    final_name_str = convertTuple(final_name)
  
    for i in range(res.shape[-1]): # Loop over channels
        res_path = final_name_str.format(i+1)
        skimage.io.imsave(res_path , res[..., i])