assert __name__ == "__main__"
"""
    Tool to visualize a solution file. Images will be saved out in the current folder.
    Use and modify as you want
    Usage:
        python3 visualize.py /data/train /data/solution.csv
"""

import sys
import pickle
import os
import math
import random
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
import cv2

def explore_sequence(seq):
    print(seq["ID"])
    # number of images
    numImg = len(seq["path"]) 
    print("Images: "+str(numImg))

    width = 1024
    height = 1024

    # Create 3D data cube to hold data, assuming all data have
    # array sizes of 1024x1024 pixels.
    data_cube = np.empty((width,height,numImg))

    for i in range(numImg):
        # read image and header from FITS file
        img, hdr = fits.getdata(seq["path"][i], header=True)
        
        # Normalize by exposure time (a good practice for LASCO data)
        img = img.astype('float64') / hdr['EXPTIME']
        
        # Store array into datacube (3D array)
        data_cube[:,:,i] = img - medfilt2d(img, kernel_size=9)

    rdiff = np.diff(data_cube, axis=2)
    print(seq["truth"])

    for i in range(numImg-1):

        medsub = -rdiff[:,:,i]
        medsub = cv2.min(medsub, 10.)
        medsub = cv2.max(medsub, -10.)
        medsub = (medsub + 10.) * 255. / 20.
        medsub = np.uint8(medsub)
        (T, mask) = cv2.threshold(medsub, 190, 255, cv2.THRESH_BINARY)
        medsub = cv2.bitwise_and(medsub, mask)

        # mask out sun
        cv2.circle(medsub, (512,512), 190, (0,0,0), cv2.FILLED)

        blobs_log = blob_log(medsub, max_sigma=5, min_sigma=1, num_sigma=5, threshold=0.1)
        if len(blobs_log)>0:
            blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        for blob in blobs_log:
            y, x, r = blob
            cv2.circle(medsub, (int(float(x)),int(float(y))), int(r)+3, (255,255,255), 1)

        # Draw lines
        for j in range(numImg-1):
            if seq["images"][j] in seq["truth"]:
                xy1 = seq["truth"][seq["images"][j]]
                if seq["images"][j+1] in seq["truth"]:
                    xy2 = seq["truth"][seq["images"][j+1]]
                    cv2.line(medsub, (int(float(xy1[0])),int(float(xy1[1]))), (int(float(xy2[0])),int(float(xy2[1]))), (255,255,255))

        if seq["images"][i] in seq["truth"]:
            xy = seq["truth"][seq["images"][i]]
            cv2.circle(medsub, (int(float(xy[0])),int(float(xy[1]))), 10, (255,255,255), 1)
        cv2.imwrite(seq["ID"]+"_"+str(i)+".png", medsub)

# folder for the set
folder_in = sys.argv[1]
# ground truth file to be visualized
comet_filename = sys.argv[2]

data_set = []
with open(comet_filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        tokens = line.split(',')
        seq = {}
        seq["ID"] = tokens[0]
        images = []
        paths = []
        truths = {}
        for i in range( (len(tokens)-2)//3 ):
            images.append(tokens[1+i*3])
            paths.append(os.path.join(folder_in, tokens[0], tokens[1+i*3]))
            truths[tokens[1+i*3]] = [float(tokens[2+i*3]),tokens[3+i*3]]
        images.sort()
        paths.sort()
        seq["images"] = images
        seq["path"] = paths
        seq["truth"] = truths
        if len(images)>0:
            data_set.append(seq)

for s in data_set:
    explore_sequence(s)
 