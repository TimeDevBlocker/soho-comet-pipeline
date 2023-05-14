#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2021

PURPOSE: 
    This code provides two very basic image processing algorithms that
are effective for removing the excess background signal (known as the "focorona")
from raw LASCO data. These are established, publicly released algorithms that are 
well-known to the heliophysics community. 

Details of the two enclosed algorithms - "median subtraction" and "running difference"
are provided as notes/comments in the code below.

USAGE:
    Place a series (3 or more) of LASCO C2 .fts files into a 'C2DATA' folder
    Run
        >>> from c2_processing import c2_process
        >>> c2_process( )  
         - or -
        >>> c2_process( rundiff=True )
OUTPUT:
    - A series of processed png files will be created in the 'C2DATA' folder  
    
*** Please note: ***
There is very minimal fault-tolerance in this code. It is provided simply to 
illustrate the basic algorithms, but does not perform any checks for bad, missing,
or incorrectly shaped data. There exist many different methods/packages/functions 
to achieve the same, or similar, results as shown here, with differing levels of 
efficiency. 

DEPENDENCIES / REQUIRED PACKAGES:
    matplotlib
    numpy
    scipy
    astropy
    datetime
    glob

@author: Dr. Karl Battams
"""

import matplotlib.pyplot as plt
import glob
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
import datetime
import torch
import torch.nn.functional as F
import os

########################################################################
# Function to perform median subtraction and save resulting files as PNGs
########################################################################

def med_subtract( indata, dst_path, dateinfo, kern_sz = 25 , annotate_words=True):
    dpi = 80            # dots per inch
    width = 1024        # image size (assuming 1024x1024 images)
    height = 1024       # image size (assuming 1024x1024 images)
    kern = kern_sz       # size of smoothing kernel <- User defined

    imgmin = -3.    # MAX value for display <- User defined
    imgmax = 3.     # MIN value for display <- User defined

    figsize = width / float(dpi), height / float(dpi) # Set output image size
    numfiles = indata.shape[2] # For loop counter
    
    """  ## Note about median subtraction  ##
    The process of median subtraction uses medfilt2d to create a 'smoothed' version of the
    image and then subtracts that smoothed image from the original, effectively operating 
    like a high-pass filter. The size of the smoothing kernel (default = 25) is variable.
    """
    plt.ioff()                      # Turn off interactive plotting

    # Create images 
    for i in range( numfiles ):
        outname = dst_path + dateinfo[i].strftime('%Y%m%d_%H%M_C2_medfilt.png')
        if os.path.exists(outname):
            continue
        print("Writing image %i of %i with median-subtraction" % (i+1,numfiles))
        medsub = indata[:,:,i] - medfilt2d(indata[:,:,i], kernel_size=kern) # Apply filter; see note above
        # The following commands just set up a figure with no borders, and writes the image to a png.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(np.fliplr(medsub), vmin=imgmin,vmax=imgmax,cmap='gray', interpolation='nearest',origin='lower')
        if annotate_words:
            ax.annotate(dateinfo[i].strftime('%Y/%m/%d %H:%M'), xy=(10,10), xytext=(320, 1010),color='cyan', size=30, ha='right')
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        fig.savefig(outname, dpi=dpi, transparent=True)
        plt.close()
        
    print("Median filtering process complete.")
    return 1
  

########################################################################
# Function to perform running difference and save resulting files as PNGs  
########################################################################

def rdiff( indata, dateinfo ):
    
    """  ## Note about running difference  ##
    The process of running difference involves subtracting the previous image
    in a sequence from the current image. This process removes static structures
    but emphasizes features in motion.
    """
    
    #Perform running difference. See note above.
    rdiff = np.diff(indata, axis=2)
    
    # truncate date information as running difference "loses" the first image
    dateinfo = dateinfo[1:]
    
    # Write PNGS files
    dpi = 80            # dots per inch
    width = 1024        # image size (assuming 1024x1024 images)
    height = 1024       # image size (assuming 1024x1024 images)
    imgmin = -3.        # MAX value for display <- User defined
    imgmax = 3.         # MIN value for display <- User defined

    figsize = width / float(dpi), height / float(dpi) # Set output image size
    numfiles = rdiff.shape[2]       # For loop counter
    
    plt.ioff()                  # Turn off interactive plotting
    # Create images
    for i in range( numfiles ):
        print("Writing image %i of %i with running-difference processing" % (i+1,numfiles))
        
        # The following commands just set up a figure with no borders, and writes the image to a png.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(np.fliplr(rdiff[:,:,i]), vmin=imgmin,vmax=imgmax,cmap='gray', interpolation='nearest',origin='lower')
        ax.annotate(dateinfo[i].strftime('%Y/%m/%d %H:%M'), xy=(10,10), xytext=(320, 1010),color='cyan', size=30, ha='right')
        outname='./C2DATA/'+dateinfo[i].strftime('%Y%m%d_%H%M_C2_rdiff.png')
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
        fig.savefig(outname, dpi=dpi, transparent=True)   
        plt.close()
        
    print("Running Difference process complete.")
    return 1

###################################################################################
# MAIN (control) routine to read/prepare data and call desired processing algorithm
###################################################################################


def c2_process(fts_path, dst_path, rundiff=False, annotate_words=True, batch_mode=False):

    os.makedirs(dst_path) if not os.path.isdir(dst_path) else None
    dst_path_list = os.listdir(dst_path)
    if not batch_mode:
        for file in dst_path_list:
            os.remove(dst_path + file)

    # Gather list of file names
    lasco_files = sorted(glob.glob('{}*.fts'.format(fts_path)))
    
    # number of files
    nf = len(lasco_files) 
    
    # Create 3D data cube to hold data, assuming all LASCO C2 data have
    # array sizes of 1024x1024 pixels.
    data_cube = np.empty((1024,1024,nf))
    
    # Create an empty list to hold date/time values, which are later used for output png filenames
    dates_times = []
    
    for i in range(nf):
        # read image and header from FITS file
        img,hdr = fits.getdata(lasco_files[i], header=True)
        
        # Normalize by exposure time (a good practice for LASCO data)
        img = img.astype('float64') / hdr['EXPTIME']

        if img.shape[0] == 512 and img.shape[1] == 512:
            img = torch.from_numpy(img)
            img = F.interpolate(img[None, None, :, :], size=(1024, 1024), mode='bilinear')[0, 0, :, :]
            img = img.numpy()

        data_cube[:, :, i] = img
        
        # Retrieve image date/time from header; store as datetime object
        dates_times.append( datetime.datetime.strptime( hdr['DATE-OBS']+' '+hdr['TIME-OBS'], '%Y/%m/%d %H:%M:%S.%f') )
    print('processing med_subtract...')
    # Call processing routine. Defaults to median subtraction.    
    if rundiff:
        _ = rdiff( data_cube, dates_times )
    else:
        _ = med_subtract( data_cube, dst_path, dates_times , annotate_words=annotate_words)