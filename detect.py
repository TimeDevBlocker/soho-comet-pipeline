"""
File: detect.py
Note: This code generates detections from raw image data
Date: 2022-02-26
Author: D. Parrott
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
from astropy.time import Time
from scipy.signal import medfilt2d
from scipy import ndimage
from numba import jit
from numba import njit
from numba import typed
from numba import types
import matplotlib.pyplot as plt
import cv2
import faulthandler; faulthandler.enable()
import multiprocessing
import warnings
import track_quality

warnings.filterwarnings("ignore")


# Define some constants.
SOHO_NUM_DELTAS_AWAY=2.75
SOHO_IMG_STATISTIC_OFFSET=150
THRESHOLD_GRID_SIZE=32


# Class definitions
class MyFITSImg:
    pass

class MyTrack:
    pass

class MyDetect:
    pass

class MySeq:
    pass

# ComputeImgMedian
# Input:
#   img: A given image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
# Output:
#   Image background level
def ComputeImgMedian(img,width,height):
    off = SOHO_IMG_STATISTIC_OFFSET
    listValues=[]
    j = off
    while (j<height-off):
        i = off
        listValues.append(img[j][i])
        i = width - off
        listValues.append(img[j][i])
        j = j + 1

    i = off
    while (i<width-off):
        j = off
        listValues.append(img[j][i])
        j = height - off
        listValues.append(img[j][i])
        i = i + 1

    medianVal = np.median(listValues)
    return medianVal

# ComputeNormalizationFactor
# Input:
#   data_cube: A list of images
#   width: Image width, in pixels
#   height: Image height, in pixels
# Output:
#   normalizationFactor: Median background level
#   listBkgndLevels: List of image background levels
def ComputeNormalizationFactor(data_cube,width,height):
    listBkgndLevels=[]
    slen = data_cube.shape[2]
    for i in range(slen):
        bkgndLevel = ComputeImgMedian(data_cube[:, :, i], width, height)
        listBkgndLevels.append(bkgndLevel)

    normalizationFactor = np.median(listBkgndLevels)
    return (normalizationFactor, listBkgndLevels)

# ScaleImages
# Input:
#   data_cube: A list of images
#   width: Image width, in pixels
#   height: Image height, in pixels
#   normalizationFactor: The median background level
# Modifies:
#   Pixel values of each image so that each image achieves
#     a background level comparable to the median background level.
def ScaleImages(data_cube, width, height, normalizationFactor):
    slen = data_cube.shape[2]
    for i in range(slen):
        scaleFactor = normalizationFactor[0] / max(1, normalizationFactor[1][i])
        data_cube[:,:,i] = np.multiply(data_cube[:, :, i], scaleFactor)

# SubtractStackFromImages
# Input:
#   data_cube: A list of images
#   stacked_img: An image created from a stack of images
#   width: Image width, in pixels
#   height: Image height, in pixels
# Modifies:
#   Pixel values of each image are modified as follows:
#   NEW_PIXEL_VALUE = IMG_PIXEL_VALUE - STACKED_IMG_PIXEL_VALUE
def SubtractStackFromImages(data_cube, stacked_img, width, height):
    slen = data_cube.shape[2]
    for i in range(slen):
        data_cube[:,:,i] = np.subtract(data_cube[:,:,i], stacked_img)

# ExtractFlux
# Input:
#   img: An image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
# Output:
#   A new image after having been convolved with a 3x3 window
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def ExtractFlux(img,width,height):

    # Create a new image, initialized to 0
    out = np.full_like(img, 0)

    pk_val=0
    sum=0
    v1=0
    a=0
    b=0
    i=0
    j=0
    p=0
    q=0

    j=0
    while (j<height):
        i=0
        while (i<width):

            pk_val=0
            sum=0

            b=0
            while (b<3):
                a=0
                while (a<3):
                    p = i-1+a
                    q = j-1+b

                    if (p<0 or p>=width or q<0 or q>=height):
                        a=a+1
                        continue

                    v1 = img[q][p]

                    if (v1 > pk_val):
                        pk_val=v1

                    sum += v1
                    a=a+1

                b=b+1
            sum -= pk_val

            if (sum < 0):
                sum = 0

            if (sum > 65535):
                sum = 65535

            out[j][i] = sum

            i = i + 1

        j = j + 1
    return out

# ComputeStatistics
# Input:
#   listValues: A list of values
# Output:
#   val_25: The 25% order statistic
#   val_50: The 50% order statistic
#   val_75: The 75% order statistic
#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
#@jit(complex128[:](float64,float64[:],float64))
#@jit('Tuple((int64,int64,int64))(int64[:])',nopython=True)
#@njit()
@jit(nopython=True)
def ComputeStatistics(listValues):

    val_25 = 0
    val_50 = 0
    val_75 = 0

    # numba does not work with numpy sort
    #listValues = np.sort(listValues)
    listValues.sort()
    slen = len(listValues)

    idx_25 = int(0.25 * slen)
    idx_50 = int(0.50 * slen)
    idx_75 = int(0.75 * slen)

    for i in range(slen):

        if (idx_25 == i):
            val_25 = listValues[i]

        if (idx_50 == i):
            val_50 = listValues[i]

        if (idx_75 == i):
            val_75 = listValues[i]

    return (val_25, val_50, val_75)

# CreateThresholdMap
# Input:
#   img: A given image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
#   grid_size: Size of grid cells, in pixels
# Output:
#   A new image, with pixels set to respective threshold values
@jit(nopython=True)
def CreateThresholdMap(img, width, height, grid_size):

    # Create a new image, initialized to 0
    out = np.full_like(img, 0)

    j=0
    while (j<height):

        i=0
        while (i<width):

            listValues = []

            b=0
            while (b<grid_size):
                a=0
                while (a<grid_size):

                    p = i + a
                    q = j + b

                    if (p<0 or p>=width or q<0 or q>=height):
                        a = a + 1
                        continue

                    v1 = img[q][p]
                    listValues.append(v1)

                    a = a + 1

                b = b + 1

            stats = ComputeStatistics(listValues)
            fPct25 = stats[0]
            fPct50 = stats[1]
            fPct75 = stats[2]

            fDelta = fPct75 - fPct25
            fThreshold = fPct50 + SOHO_NUM_DELTAS_AWAY * (fDelta)

            nThreshold = int(fThreshold + 0.5)

            if (nThreshold < 0):
                nThreshold = 0

            if (nThreshold > 65535):
                nThreshold = 65535

            b=0
            while (b<grid_size):
                a=0
                while (a<grid_size):

                    p = i + a
                    q = j + b

                    if (p<0 or p>=width or q<0 or q>=height):
                        a = a + 1
                        continue

                    out[q][p] = nThreshold

                    a = a + 1

                b = b + 1

            i = i + grid_size

        j = j + grid_size
    return out

# InterpolateMap
# Input:
#   img: A given image to evaluate
#   width: Image width in pixels
#   height: Image height in pixels
#   grid_size: Size of grid cells, in pixels
# Output:
#   A modified threshold map, selecting maximum threshold from adjacent cells
@jit(nopython=True)
def InterpolateMap(img, width, height, grid_size):

    # Create a new image, initialized to 0
    out = np.full_like(img, 0)

    j=0
    while (j<height):

        i=0
        while (i<width):

            bin_i = int(i / grid_size)
            bin_j = int(j / grid_size)

            pk_val = 0

            b=0
            while (b<3):

                a=0
                while (a<3):

                    p = bin_i - 1 + a
                    q = bin_j - 1 + b

                    p *= grid_size
                    q *= grid_size

                    if (p<0 or p>=width or q<0 or q>=height):
                        a = a + 1
                        continue

                    v1 = img[q][p]

                    if (v1 > pk_val):
                        pk_val=v1

                    a = a+1

                b = b+1

            out[j][i] = pk_val
            i = i+1
        j = j+1

    return out

# FilterSinglePixels
# Input:
#   img: A given image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
# Modifies:
#   Image pixels are set to 0 if there are no adjacent pixels with non-zero values
@jit(nopython=True)
def FilterSinglePixels(img,width,height):

    j=0
    while (j<height):
        i=0
        while (i<width):
            nCount=0

            b=0
            while (b<3):
                a=0
                while (a<3):
                    p = i - 1 + a
                    q = j - 1 + b

                    if (p<0 or p>=width or q<0 or q>=height):
                        a = a + 1
                        continue

                    v1 = img[q][p]

                    if (v1 > 0):
                        nCount = nCount + 1

                    a = a+1

                b=b+1

            if (nCount < 2):
                img[j][i] = 0

            i = i+1
        j = j+1

# GetBounds
# Input:
#   img: A given image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
#   i: Current x-coordinate
#   j: Current y-coordinate
#   visited: A 2d array indicating which pixels have been visited
#   res: A result structure comprised of counts, and the upper-left and lower-right corner locations
# Output:
#   res: A result structure comprised of counts, and the upper-left and lower-right corner locations
@jit(nopython=True)
def GetBounds(img, width, height, i, j, visited, res):

    if (i<0 or i>=width or j<0 or j>=height):
        return res

    if (img[j][i] <= 0):
        return res

    if (visited[j][i] > 0):
        return res

    # Mark the pixel as having been visited
    visited[j][i] = 1

    # Increment the count of consolidated pixels
    res[0] = res[0] + 1

    if (i < res[1]):
        res[1] = i
    if (i > res[3]):
        res[3] = i
    if (j < res[2]):
        res[2] = j
    if (j > res[4]):
        res[4] = j

    res = GetBounds(img, width, height, i-1, j-1, visited, res)
    res = GetBounds(img, width, height, i+1, j-1, visited, res)
    res = GetBounds(img, width, height, i+1, j+1, visited, res)
    res = GetBounds(img, width, height, i-1, j+1, visited, res)

    res = GetBounds(img, width, height, i+0, j-1, visited, res)
    res = GetBounds(img, width, height, i+1, j+0, visited, res)
    res = GetBounds(img, width, height, i+0, j+1, visited, res)
    res = GetBounds(img, width, height, i-1, j+0, visited, res)

    return res

# ComputeCentroid
# Input:
#   img: A given image to be evaluated
#   width: Image width in pixels
#   height: Image height in pixels
#   x1: The X-coordinate of the upper-left bounds of the object
#   y1: The Y-coordinate of the upper-left bounds of the object
#   x2: The X-coordinate of the lower-right bounds of the object
#   y2: The Y-coordinate of the lower-right bounds of the object
# Output:
#   nCtrX: The X-coordinate of the object center
#   nCtrY: The Y-coordinate of the object center
@jit(nopython=True)
def ComputeCentroid(img, width, height, x1, y1, x2, y2):

    fCountX = 0;
    fCountY = 0;
    fSumX = 0;
    fSumY = 0;

    j = y1
    while (j<=y2):

        i = x1
        while (i <= x2):

            if (i<0 or i>=width or j<0 or j>=height):
                i = i + 1
                continue

            v1 = img[j][i]

            fSumX += i * v1
            fCountX += v1

            fSumY += j * v1
            fCountY += v1

            i = i + 1

        j = j + 1

    fCtrX = fSumX / max(1, fCountX)
    fCtrY = fSumY / max(1, fCountY)

    nCtrX = int(fCtrX + 0.5)
    nCtrY = int(fCtrY + 0.5)

    if (nCtrX<0):
        nCtrX=0

    if (nCtrX>=width):
        nCtrX = width-1

    if (nCtrY<0):
        nCtrY=0

    if (nCtrY>=height):
        nCtrY = height-1

    return (nCtrX, nCtrY)

# ConsolidatePixels
# Input:
#   img: The image to evaluate
#   width: Image width, in pixels
#   height: Image height, in pixels
# Output:
#   A new image, after having performed the consolidation routine.
@jit(nopython=True)
def ConsolidatePixels(img, width, height):

    # Create a new image, initialized to 0
    out = np.full_like(img, 0)

    # Create a map to indicate visited cells, initialized to 0
    visited = np.full_like(img, 0)

    j=0
    while (j<height):
        i=0
        while (i<width):

            v1 = img[j][i]

            if (v1 <= 0):
                i = i + 1
                continue

            if (visited[j][i] > 0):
                i = i + 1
                continue

            nCount = 0
            x1 = i
            y1 = j
            x2 = i
            y2 = j

            res = [nCount, x1, y1, x2, y2]

            res = GetBounds(img, width, height, i, j, visited, res)

            nCount = res[0]
            x1 = res[1]
            y1 = res[2]
            x2 = res[3]
            y2 = res[4]

            centroid = ComputeCentroid(img, width, height, x1, y1, x2, y2)

            ctr_x = centroid[0]
            ctr_y = centroid[1]

            if (nCount < 0):
                nCount = 0

            if (nCount > 65535):
                nCount = 65535

            out[ctr_y][ctr_x] = nCount

            i = i + 1

        j = j + 1

    return out

# AllocEmptyMapList
# Input:
#   w: Map horizontal dimensions
#   h: Map vertical dimensions
# Output:
#   A map of empty detection lists.
def AllocEmptyMapList(w, h):
    return [ [ [] for i in range(w) ] for j in range(h) ]

# GenerateMapListDetections
# Input:
#   img: An image to be evaluated
#   width: Image width, in pixels
#   height: Image height, in pixels
#   grid_size: Size of grid cells, in pixels
# Output:
#   A map of detection lists, each populated with detections in their respective cells.
def GenerateMapListDetections(img, width, height, grid_size):

    w = int(width / grid_size) + 1
    h = int(height/ grid_size) + 1

    map_detect_list = AllocEmptyMapList(w, h)

    j=0
    while (j<height):

        i = 0
        while (i<width):

            v1 = img[j][i]

            if (v1 <= 0):
                i = i+1
                continue

            ii = int(i/grid_size)
            jj = int(j/grid_size)

            map_detect_list[jj][ii].append((i,j))

            i = i + 1
        j = j + 1

    return map_detect_list

# PrintMapListDetections
# Input:
#   map_detections: A map of detections to be printed
#   width: Image width in pixels
#   height: Image height in pixels
#   grid_size: Size of grid cells, in pixels
# Output:
#   Map detections printed to stdout
def PrintMapListDetections(map_detections, width, height, grid_size):

    w = int(width / grid_size) + 1
    h = int(height / grid_size) + 1

    j = 0
    while (j<h):
        i=0
        while (i<w):

            print("Grid: ("+repr(i)+", "+repr(j)+")")

            slen = len(map_detections[j][i])

            for t in range(slen):
                detect = map_detections[j][i][t]
                print(" " +repr(detect[0])+", "+repr(detect[1]))

            print("")

            i = i + 1

        j = j + 1

# CreateDetectionsForImage
# Input:
#   img: An image to be evaluated
#   img_list: List of images in the current sequence
#   img_idx: The index of the image being evaluated
# Modifies:
#   img_detections: An image comprised of detection values
#   map_detections: A map of detections, for faster lookup
def CreateDetectionsForImage(img, img_list, img_idx):

    width = img_list[0].width
    height = img_list[0].height

    #print("Before Extract")
    # Compute a 3x3 flux measurement for each pixel
    img_flux = ExtractFlux(img, width, height)
    #print("After extract")
    #print("img_flux: value(800,200)="+repr(img_flux[200][800]))

    img_threshold = CreateThresholdMap(img_flux, width, height, THRESHOLD_GRID_SIZE)
    #print("img_threshold: value(800,200)="+repr(img_threshold[200][800]))
    img_pk_thresh = InterpolateMap(img_threshold, width, height, THRESHOLD_GRID_SIZE)
    #print("img_pk_thresh: value(800,200)="+repr(img_pk_thresh[200][800]))

    # Subtract the threshold from the flux image
    img = np.subtract(img_flux, img_pk_thresh)

    # Truncate to zero
    img[img<0]=0

    # Filter single pixels
    FilterSinglePixels(img, width, height)

    # Consolidate pixels into single detections
    img_list[img_idx].img_detections = ConsolidatePixels(img, width, height)

    #print("img_detections(923, 415)="+repr(img_list[img_idx].img_detections[415][923]))

    img_list[img_idx].map_detections = GenerateMapListDetections(img_list[img_idx].img_detections,
                                                                 width, height, THRESHOLD_GRID_SIZE)

    # if (0==img_idx):
    #     PrintMapListDetections(img_list[img_idx].map_detections, width, height, THRESHOLD_GRID_SIZE)

# CreateDetections
# Input:
#   data_cube: A list of images
#   img_list: List of images in the current sequence
# Modifies:
#   Generates detections for each image.
def CreateDetections(data_cube, img_list):
    slen = data_cube.shape[2]
    for i in range(slen):
        CreateDetectionsForImage(data_cube[:,:,i], img_list, i)
