"""
File: test.py
Note: This is the main driver code to load images and generate results
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
import detect
import tracker

warnings.filterwarnings("ignore")




# Define some constants.
SOHO_MAX_TRACKS=250000
SOHO_MAX_SEC_PER_PIXEL=140
SOHO_MIN_SEC_PER_PIXEL=36  # 44
SOHO_FIT_ORDER_CUTOFF=0.97
SOHO_FIT_ORDER=1
SOHO_NUM_TRACKS_OUTPUT=2
SOHO_PCT_SAME_DETECTS=0.60
THRESHOLD_GRID_SIZE=32
R2D=(180 / 3.1415926535)

# Class definitions
class MyFITSImg:
    pass

class MyTrack:
    pass

class MyDetect:
    pass

class MySeq:
    pass

# normalizeDate
# Input: A string formatted YYYY/MM/DD HH:MM:SS.sss
# Output: A string formatted YYYY-MM-DD HH:MM:SS.sss
def normalizeDate(date):
    newDate = '-'.join(str.zfill(elem,2) for elem in date.split('/'))
    return newDate

# compute_elapsed_time_in_sec2
# Input: Two time objects
# Output: Elapsed time, floating point, in seconds.
def compute_elapsed_time_in_sec2(dtime_a, dtime_b):
    delta_time = (dtime_b - dtime_a)
    dt_float = delta_time.to_value('sec', subfmt='float')
    return dt_float

# Enable to see debug info
DEBUG = False

# GenerateResultString2
# Input:
#   seq_full: A full image sequence
#   track: A given track object
#   subset_offset: The offset associated with the current image subset
#   listDateObs_full: List of DateObs for the full image sequence
# Output:
#   A string containing the specified track information and quality score
def GenerateResultString2(seq_full, track, subset_offset, listDateObs_full):

    num_img = len(seq_full["images"])
    result = seq_full["ID"] + ","

    fQuality = track.fGlobalQuality

    slen = len(listDateObs_full)
    fDateObs_First = listDateObs_full[0]
    fDateObs_Last = listDateObs_full[slen-1]

    fDeltaTimeInSec = fDateObs_Last - fDateObs_First
    fDeltaTimeInHours = fDeltaTimeInSec / 3600.0

    if (fDeltaTimeInHours > 24.0 and slen>40):
        # This is considered a 'long' dataset (spanning more than 24 hours)
        # Compensate for the likelihood that track positions will be inaccurate beyond that time.
        fQuality *= 0.25

    for t in range(num_img):
        xy = tracker.ComputeTrackPosition(track, t-subset_offset, listDateObs_full[t])
        x = int(xy[0]+0.5)
        y = int(xy[1]+0.5)
        imgid = seq_full["path"][t].split("/")[-1]
        result += imgid + "," + repr(x) + "," + repr(y) + ","

    result += repr(fQuality) + "\n"

    return result

# GenerateResultString
# Input:
#   seq_full: A full image sequence
#   listTracks: A list of tracks eligible for output to the result file
#   subset_offset: The offset associated with the current image subset
#   listDateObs_full: List of DateObs for the full image sequence
# Output:
#   The requested tracks for output to the result file
def GenerateResultString(seq_full, listTracks, subset_offset, listDateObs_full):

    slen = len(listTracks)

    result = ""

    for t in range(slen):
        if (t>=SOHO_NUM_TRACKS_OUTPUT):
            break

        cur_track = listTracks[t]
        result += GenerateResultString2(seq_full, cur_track, subset_offset, listDateObs_full)

    return result

# explore_sequence
# Input:
#   seq_full: The full image sequence
#   seq: The current image sequence
#   subset_offset: The offset associated with the current image subset
#   listDateObs_full: List of DateObs for the full image sequence
# Output:
#   A result string to be output to the result file
def explore_sequence(seq_full, seq, subset_offset, listDateObs_full):
    """
        Extract the comets from a given sequence
    """
    if DEBUG:
        print("Sequence: " + seq["ID"])
    # number of images
    numImg = len(seq["path"]) 
    if DEBUG:
        print("Number of images: "+str(numImg))

    img_list = []

    width = 1024
    height = 1024

    # Create 3D data cube to hold data, assuming all data have
    # array sizes of 1024x1024 pixels.
    data_cube = np.empty((width,height,numImg))

    timestamps = [0] * numImg


    depoch = Time('1970-01-01T00:00:00.0', scale='utc', format='isot')

    for i in range(numImg):


        img = MyFITSImg()
        img.fullpath = seq["path"][i]
        img.hdulist = fits.open(img.fullpath)
        img.width = img.hdulist[0].header['NAXIS1']
        img.height = img.hdulist[0].header['NAXIS2']
        img.datestring = img.hdulist[0].header['DATE-OBS']
        img.timestring = img.hdulist[0].header['TIME-OBS']
        my_date_str = img.datestring + ' ' + img.timestring
        newDate = normalizeDate(my_date_str)
        dtime = Time(newDate, scale='utc', format='iso')
        img.fDateObs = compute_elapsed_time_in_sec2(depoch, dtime)
        img.data = img.hdulist[0].data
        img.img_detections = []
        img.map_detections = []
        img_list.append(img)

        #ymd = epoch_seconds_to_gregorian_date(img.fDateObs)
        #print(repr(i)+") "+repr(ymd[0])+"-"+repr(ymd[1])+"-"+repr(ymd[2]))
        #print("subset_offset="+repr(subset_offset))

        #if (i>0):
            #delta_time = img_list[i].fDateObs - img_list[i-1].fDateObs
            #print("delta_time="+repr(delta_time))

    for i in range(numImg):
        # read image and header from FITS file
        img, hdr = fits.getdata(seq["path"][i], header=True)
        
        # Collect timestamps
        timestamps[i] = hdr["MID_TIME"]
        
        # Store array into datacube (3D array)
        data_cube[:,:,i] = img

    # Floating point not desired -- convert to integer units
    data_cube = data_cube.astype(int)

    # Compute a normalization factor by which the images will be multiplied.
    normalizationFactor = detect.ComputeNormalizationFactor(data_cube, width, height)

    # Scale each image by the normalization factor
    detect.ScaleImages(data_cube, width, height, normalizationFactor)

    # Compute an average stack of the images.
    StackImgZeroMotion = np.mean(data_cube, axis=2)

    # Subtract the average stack from each image.
    detect.SubtractStackFromImages(data_cube, StackImgZeroMotion, width, height)

    # Truncate each value to 0 (no negatives)
    data_cube[data_cube < 0] = 0

    # Generate detections from each image
    detect.CreateDetections(data_cube, img_list)

    # Generate tracks from the detections
    listTracks = tracker.CreateTracks(img_list, width, height, THRESHOLD_GRID_SIZE)

    listDateObs=[]
    listImgDetections=[]
    slen = len(img_list)
    for t in range(slen):
        listDateObs.append(img_list[t].fDateObs)
        listImgDetections.append(img_list[t].img_detections)

    #print("(BEFORE) Num tracks="+repr(len(listTracks)))

    if (len(listTracks)>SOHO_MAX_TRACKS):
        return ""

    # Reduce tracks
    listTracks = tracker.ReduceTracks(listTracks, img_list, listDateObs, listImgDetections, width, height)
    listTracks = tracker.CullToTopNTracks(listTracks, 50)
    listTracks = tracker.ConsolidateTracks(listTracks, numImg)
    tracker.FinalizeMotion(listTracks, numImg)

    #print("(AFTER) Num tracks=" + repr(len(listTracks)))

    #PrintTracks(listTracks, img_list)

    result = GenerateResultString(seq_full, listTracks, subset_offset, listDateObs_full)

    return result

# GenerateSequences
# Input:
#   seq_full: The full image sequence
# Output:
#   A list of sequences to be explored
def GenerateSequences(seq_full):

    listSequences = []
    numImg = len(seq_full["path"])

    depoch = Time('1970-01-01T00:00:00.0', scale='utc', format='isot')

    prev_dateobs = 0
    delta_time = 0
    max_delta_time = 6*3600
    max_img_per_seq = 50
    min_img_per_seq = 5

    cur_seq = MySeq()
    cur_seq.seq = {"ID": "none", "images": [], "path": []}
    cur_seq.subset_offset = 0

    cur_num_img = 0

    for t in range(numImg):

        fullpath = seq_full["path"][t]
        hdulist = fits.open(fullpath)
        datestring = hdulist[0].header['DATE-OBS']
        timestring = hdulist[0].header['TIME-OBS']
        my_date_str = datestring + ' ' + timestring
        newDate = normalizeDate(my_date_str)
        dtime = Time(newDate, scale='utc', format='iso')
        fDateObs = compute_elapsed_time_in_sec2(depoch, dtime)

        if (0==t):
            prev_dateobs = fDateObs

        delta_time = fDateObs - prev_dateobs

        if (delta_time < max_delta_time and cur_num_img < max_img_per_seq):
            cur_num_img += 1
            cur_seq.seq["ID"] = seq_full["ID"]
            cur_seq.seq["images"].append(seq_full["images"][t])
            cur_seq.seq["path"].append(seq_full["path"][t])
        else:
            # Either the delta time (gap) has been exceeded
            #  or there are too many images for the current sequence

            # Add the current sequence to the list of sequences
            if (cur_num_img>=min_img_per_seq):
                listSequences.append(cur_seq)

            # Construct new sequence
            cur_seq = MySeq()
            cur_seq.seq = {"ID": seq_full["ID"], "images": [seq_full["images"][t]], "path": [seq_full["path"][t]]}
            cur_seq.subset_offset = t

            cur_num_img = 1

        prev_dateobs = fDateObs

    # Add any remaining sequence
    if (cur_num_img>=min_img_per_seq):
        listSequences.append(cur_seq)

    return listSequences

# GenerateListDateObs
# Input:
#   seq: A given sequence of images
# Output:
#   A list of DateObs pertaining to the given image sequence
def GenerateListDateObs(seq):

    depoch = Time('1970-01-01T00:00:00.0', scale='utc', format='isot')
    listDateObs = []
    numImg = len(seq["images"])

    for t in range(numImg):
        fullpath = seq["path"][t]
        hdulist = fits.open(fullpath)
        datestring = hdulist[0].header['DATE-OBS']
        timestring = hdulist[0].header['TIME-OBS']
        my_date_str = datestring + ' ' + timestring
        newDate = normalizeDate(my_date_str)
        dtime = Time(newDate, scale='utc', format='iso')
        fDateObs = compute_elapsed_time_in_sec2(depoch, dtime)
        listDateObs.append(fDateObs)

    return listDateObs

# process_sequence
# Input:
#   seq_full: The full image sequence to be explored
# Output:
#   The result string(s) associated with the entire image sequence
def process_sequence(seq_full):

    # Find comets in the sequence and return only the longest matched one
    result = []
    slen = len(seq_full["images"])
    if slen < 5:
        #Ignore short sequences.
        print(seq_full["progress"])
        return result

    track_quality.PopulateGridDirection()

    listDateObs_full = GenerateListDateObs(seq_full)
    listSequences = GenerateSequences(seq_full)

    nseq = len(listSequences)

    for t in range(nseq):

        cur_seq = listSequences[t]
        seq = cur_seq.seq
        subset_offset = cur_seq.subset_offset

        try:
            bestComet = explore_sequence(seq_full, seq, subset_offset, listDateObs_full)
            if (0!=len(bestComet)):
                result.append(bestComet)
        except Exception as e:
            print("Error: "+str(e))
            pass

    print(seq_full["progress"])
    return result


def test(folder_in, output_file):
    print('Calculate location...')
    data_set = []
    # Scan folder for all sequences
    for (dirpath, dirnames, filenames) in os.walk(folder_in):
        dirnames.sort()

        seq = {}
        cometID = os.path.relpath(dirpath, folder_in)
        seq["ID"] = cometID
        images = []
        paths = []
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext == '.fts':
                images.append(filename)
                paths.append(os.path.join(dirpath, filename))

        images.sort()
        paths.sort()
        seq["images"] = images
        seq["path"] = paths
        if len(images) > 0:
            data_set.append(seq)

    for i, s in enumerate(data_set):
        s["progress"] = "Completed!" + s["ID"] + " " + str(i + 1) + "/" + str(len(data_set))

    pool = multiprocessing.Pool()
    result_async = [pool.apply_async(process_sequence, args=(s,)) for s in data_set]
    results = [r.get() for r in result_async]
    with open(output_file, 'w') as f:
        for r in results:
            if len(r) > 0:
                f.writelines(r)
                f.flush()
    print('Calculate finished.')


if __name__ == "__main__":
    #######################################################################################
    # python test.py D:\研究生\杂活\2022-7-23彗星搜索\challenge_data\cmt0030\ D:\研究生\杂活\2022-7-23彗星搜索\output\220716_c2\output.csv
    folder_in = sys.argv[1]
    output_file = sys.argv[2]




