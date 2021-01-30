# -*- coding: utf-8 -*-

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from flowdec.nb import utils as nbutils 
from skimage import io
from scipy.stats import describe
from skimage.exposure import rescale_intensity, histogram, adjust_gamma
from flowdec import data as fd_data

channels = ['CY3','FITC','DAPI']
acqs = fd_data.load_celegans()
acqs.keys()

acqs['CY3'].shape(), acqs['CY3'].dtype()

for ch in channels:
    print(' Image stats (' + ch + '):', describe(acqs[ch].data.ravel()))
    
import tensorflow as tf
from flowdec import restoration as tfd_restoration

niter = 500
algo = tfd_restoration.RichardsonLucyDeconvolver(n_dims=3).initialize()
res = {ch: algo.run(acqs[ch], niter=niter) for ch in channels}

# Stack and rescale original image and result to RGB with leading z dimension
def to_rgb(acqs):
    return np.stack([rescale_intensity(acqs[ch].data, out_range='uint8').astype('uint8') for ch in channels], axis=-1)

img_acq = to_rgb(acqs)
print('Image shape (z,y,x,c):', img_acq.shape)
print('Image dytpe:', img_acq.dtype)
print('Image stats:', describe(img_acq.ravel()))

img_res = to_rgb(res)
print('Result shape(z,y,x,c):', img_res.shape)
print('Result dtype:', img_res.dtype)
print('Result stats:', describe(img_res.ravel()))

# Show histograms of original and deconvolved images 
fig, ax = plt.subplots(1, len(channels))
fig.set_size_inches(24, 3)
for i, ch in enumerate(channels):
    ax[i].set_title(ch)
    ax[i].plot(*histogram(img_acq[..., i])[::-1], label='Original')
    ax[i].plot(*histogram(img_res[..., i])[::-1], label='Result')
    ax[i].legend()
    
# Visualization
fig, axs = plt.subplots(1, 2)
fig.set_size_inches((24, 12))

zslice = slice(45, 75)
max_values = {'CY3': 50, 'FITC': 100, 'DAPI': 150}
axs[0].imshow(img_acq[zslice].max(axis=0))
axs[0].set_title('Original')
axs[1].imshow(np.stack([
        rescale_intensity(
                img_res[zslice, ..., i].clip(0, max_values[ch]),
                out_range = 'uint8'
                ) for i, ch in enumerate(channels)
        ], axis=-1).max(axis=0))
axs[1].set_title('Deconvolved')

# Show max-z projections for each channel separately:
fig, axs = plt.subplots(len(channels), 2)
fig.set_size_inches((24, 32))

for i, ch in enumerate(channels):
    axs[i][0].imshow(img_acq[zslice, ..., i].max(axis=0))
    axs[i][0].set_title('Original ({})'.format(ch))
    axs[i][1].imshow(img_res[zslice, ..., i].clip(0, max_values[ch]).max(axis=0))
    axs[i][1].set_title('Deconvolved ({})'.format(ch))

def plot_img_slices(img, max_values=None):
    z = np.arange(zslice.start, zslice.stop, 5)
    fig, axs = plt.subplots(len(z), len(channels))
    fig.set_size_inches(len(channels)*6, len(z)*6)
    for i in range(len(z)):
        for j in range(len(acqs)):
            im = img[z[i], ..., j]
            if max_values is not None:
                im = im.clip(0, max_values[channels[j]])
            axs[i][j].imshow(im)
            axs[i][j].axis('off')
            axs[i][j].set_title('{} (z={})'.format(channels[j], z[i]))
            
plot_img_slices(img_acq)
plot_img_slices(img_res, {'CY3':40, 'FITC': 40, 'DAPI': 90})

from tifffile import imsave
# Save 5-D arrays with shape (1, num_z_planes, channels, height, width)
imsave('celegans-deconvolved.tif', np.moveaxis(img_res, -1, 1)[np.newaxis], imagej=True)
imsave('celegans-origin.tif',np.moveaxis(img_acq, -1, 1)[np.newaxis], imagej=True)





    


