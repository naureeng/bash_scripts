# deconvolution by flowdec

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from skimage import io
import numpy as np
from tifffile import imsave

img_out = io.imread("flowdec/datasets/bit_5.tif")
img = np.squeeze(img_out)
psf = io.imread("flowdec/datasets/PSF647_38z.tif")
assert img.ndim == 3
assert psf.ndim == 3

algo = fd_restoration.RichardsonLucyDeconvolver(3,pad_mode='none').initialize()
res = algo.run(fd_data.Acquisition(img,psf), niter=25).data # run deconvolution

imsave('bit_5_D.tif', res)