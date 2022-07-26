"""
File: tracker.py
Note: This code handles track creation and reduction
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

# compute_elapsed_time_in_sec2
# Input: Two time objects
# Output: Elapsed time, floating point, in seconds.
def compute_elapsed_time_in_sec2(dtime_a, dtime_b):
    delta_time = (dtime_b - dtime_a)
    dt_float = delta_time.to_value('sec', subfmt='float')
    return dt_float

# AddTrackVelocity
# Input:
#   listPixelsPerHourX: A list of DateObs:PixelsPerHourX pairs
#   listPixelsPerHourY: A list of DateObs:PixelsPerHourY pairs
#   listX: A list of DateObs:X pairs
#   listY: A list of DateObs:Y pairs
#   fDateObs: The DateObs associated with the detection velocity to be added
#   fTimeElapsedInSec: The timespan, in seconds, associated with the deltaX and deltaY values
#   fDeltaX: The deltaX value for computation of the detection velocity
#   fDeltaY: The deltaY value for computation of the detection velocity
#   detect: The detection object containing the X,Y values
# Output:
#   No object returned.
# Modifies:
#   listPixelsPerHourX: Adds new DateObs:PixelsPerHourX pair.
#   listPixelsPerHourY: Adds new DateObs:PixelsPerHourY pair.
#   listX: Adds new DateObs:X pair
#   listY: Adds new DateObs:Y pair
@njit
def AddTrackVelocity(listPixelsPerHourX, listPixelsPerHourY, listX, listY, fDateObs, fTimeElapsedInSec, fDeltaX, fDeltaY, detect):

    fX = detect[0]
    fY = detect[1]

    if (abs(fTimeElapsedInSec) > 0.1):
        fElapsedTimeInHours = fTimeElapsedInSec / 3600.0

        fPixelsPerHourX = fDeltaX / fElapsedTimeInHours
        fPixelsPerHourY = fDeltaY / fElapsedTimeInHours
    else:
        fPixelsPerHourX = 0
        fPixelsPerHourY = 0

    listPixelsPerHourX.append(types.double(fDateObs))
    listPixelsPerHourX.append(types.double(fPixelsPerHourX))

    listPixelsPerHourY.append(types.double(fDateObs))
    listPixelsPerHourY.append(types.double(fPixelsPerHourY))

    listX.append(types.double(fDateObs))
    listX.append(types.double(fX))

    listY.append(types.double(fDateObs))
    listY.append(types.double(fY))

# CreateTracks3
# Input:
#   width: Image width in pixels
#   height: Image height in pixels
#   detect1: A starting detection, from the first image in the current image pair
#   map2: The detection map from the second image in the current image pair
#   fTimeElapsedInSec: The timespan between the first and second images in the current image pair
#   listTracks: A list of candidate tracks
#   fMinDistance: The minimum allowed distance, in pixels, that an object must travel given the current image pair
#   fMaxDistance: The maximum allowed distance, in pixels, that an object can travel given the current image pair
#   idx_img1: The index associated with the first image in the pair
#   idx_img2: The index associated with the second image in the pair
#   img_list: A list of images
#   grid_size: The size of the map grid cells, in pixels
# Modifies:
#   listTracks: New candidate tracks are added
def CreateTracks3(width, height, detect1, map2, fTimeElapsedInSec, listTracks, fMinDistance, fMaxDistance, idx_img1, idx_img2, img_list, grid_size):

    i = detect1[0]
    j = detect1[1]

    w = int(width / grid_size) + 1
    h = int(height / grid_size) + 1

    c = int(i / (grid_size))
    d = int(j / (grid_size))

    b=0
    while (b<3):
        a=0
        while (a<3):
            p = c - 1 + a
            q = d - 1 + b

            if (p<0 or p>=w or q<0 or q>=h):
                a = a + 1
                continue

            slen = len(map2[q][p])

            for t in range(slen):
                detect2 = map2[q][p][t]

                fDeltaX = detect2[0] - i
                fDeltaY = detect2[1] - j

                fDistance = sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY)

                if (fDistance < fMinDistance or
                    fDistance > fMaxDistance):
                    continue

                new_track = MyTrack()
                new_track.listDetectIdentifiers = typed.List.empty_list(types.int64)
                new_track.listPixelsPerHourX = typed.List.empty_list(types.double)
                new_track.listPixelsPerHourY = typed.List.empty_list(types.double)
                new_track.listX = typed.List.empty_list(types.double)
                new_track.listY = typed.List.empty_list(types.double)
                new_track.vectorPositions = []
                new_track.vecCoeffX = []
                new_track.vecCoeffY = []
                fDateObs = img_list[idx_img2].fDateObs
                AddTrackVelocity(new_track.listPixelsPerHourX,
                                 new_track.listPixelsPerHourY,
                                 new_track.listX,
                                 new_track.listY,
                                 fDateObs,
                                 fTimeElapsedInSec,
                                 fDeltaX,
                                 fDeltaY,
                                 detect2)


                nDetectID1 = (idx_img1 * width * height) + (i + width * j)
                nDetectID2 = (idx_img2 * width * height) + (detect2[0] + (width * detect2[1]))
                new_track.fDateObsFirstConfirmedDetection = -1
                new_track.fDateObsLastConfirmedDetection = -1
                new_track.listDetectIdentifiers.append(nDetectID2)
                new_track.listDetectCounts = typed.List.empty_list(types.int64)
                new_track.fSourceImgDateObs = img_list[idx_img1].fDateObs
                new_track.first_confirmed_idx_img = 99999
                new_track.first_confirmed_x = -1
                new_track.first_confirmed_y = -1
                new_track.last_confirmed_idx_img = -1
                new_track.last_confirmed_x = -1
                new_track.last_confirmed_y = -1
                new_track.source_img_idx = idx_img1
                new_track.source_img_x = i
                new_track.source_img_y = j
                new_track.bMarkedForDeletion = False
                new_track.nNumCombinedTracks = 1
                new_track.nNumDetectsAt2 = 0
                new_track.nNumDetectsAt3 = 0
                new_track.nNumDetectsAt4 = 0
                new_track.nNumDetectsAt5 = 0
                new_track.nNumDetectsAt6 = 0
                new_track.nNumDetectsAt7 = 0
                new_track.nNumDetectsAt8 = 0
                new_track.nNumGT0 = 0
                new_track.nNumGT1 = 0
                new_track.nNumGT2 = 0
                new_track.nNumGT3 = 0
                new_track.nNumGT5 = 0
                new_track.nNumGT7 = 0
                new_track.fSunMotionVector=-1
                new_track.fDirection=-1
                new_track.fGlobalQuality = 0
                new_track.fQuality = 0
                new_track.fFit_R2 = 0
                new_track.median_delta_x = 0
                new_track.median_delta_y = 0
                new_track.median_x = 0
                new_track.median_y = 0
                new_track.nMonthIndex = -1
                new_track.fVelocity = 0
                new_track.bFlaggedSunMotionVector = False
                new_track.bFlaggedDirection = False
                new_track.bFlaggedGridSection = False
                new_track.bFlaggedVelocity = False
                new_track.bFlaggedGridDirection = False
                listTracks.append(new_track)

            a = a + 1
        b = b + 1

# CreateTracks2:
# Input:
#   map1: The detection map for the first image in the current image pair
#   map2: The detection map for the second image in the current image pair
#   width: The image width, in pixels
#   height: The image height, in pixels
#   fTimeElapsedInSec: The timespan between the two images, in seconds.
#   listTracks: The current list of track candidates
#   idx_img1: The index associated with the first image in the current image pair
#   idx_img2: The index associated with the second image in the current image pair
#   img_list: The list of images for the current image sequence
#   grid_size: The size of the map grid cells, in pixels
# Modifies:
#   listTracks: New candidate tracks are added
def CreateTracks2(map1, map2, width, height, fTimeElapsedInSec, listTracks, idx_img1, idx_img2, img_list, grid_size):

    # Compute min and max distance for the given elapsed time
    fMinDistance = fTimeElapsedInSec / (SOHO_MAX_SEC_PER_PIXEL) # E.g., 140sec/pixel => 5.1"/min at 11.9"/px resolution
    fMaxDistance = fTimeElapsedInSec / (SOHO_MIN_SEC_PER_PIXEL) # E.g., 44sec/pixel => 16"/min at 11.9"/px resolution

    w = int(width / grid_size) + 1
    h = int(height / grid_size) + 1

    j = 0
    while (j<h):
        i=0
        while (i<w):

            slen = len(map1[j][i])

            for t in range(slen):
                detect1 = map1[j][i][t]

                CreateTracks3(width, height, detect1, map2, fTimeElapsedInSec, listTracks,
                             fMinDistance, fMaxDistance, idx_img1, idx_img2, img_list, grid_size)
            i = i + 1

        j = j + 1

# CreateTracks
# Input:
#   img_list: The list of images for the current image sequence
#   width: The width of the images, in pixels
#   height: The height of the images, in pixels
#   grid_size: The size of the detection map grid cells, in pixels
# Output:
#   listTracks: A list of candidate tracks for the input image list
def CreateTracks(img_list, width, height, grid_size):

    listTracks = []
    num_img = len(img_list)

    t=0
    while (t<num_img-1):

        img1 = img_list[t]
        img2 = img_list[t+1]

        fTimeElapsedInSec = img2.fDateObs - img1.fDateObs

        CreateTracks2(img1.map_detections, img2.map_detections, width, height, fTimeElapsedInSec, listTracks, t, t + 1, img_list, grid_size)

        if (len(listTracks)>SOHO_MAX_TRACKS):
            break

        t = t + 1

    return listTracks

# GetDetection2
# Input:
#   listValues: A list containing DateObs:Value pairs
#   bPrevious: When set to TRUE, initially only looks for detections that occur prior to the given timestamp
#   fDateObs_Current: The timestamp associated with the current detection
# Output:
#   fBestDateObs: The DateObs associated with the closest matching detection
#   fBestValue: The value associated with the closest matching detection
@njit
def GetDetection2(listValues, bPrevious, fDateObs_Current):

    fBestDateObs=0
    fBestValue=0
    fDateObs=0
    fValue=0

    fMinDeltaTime = 1e9
    bFound = False

    slen = len(listValues)

    t=0
    while (t<slen):
        fDateObs = listValues[t]
        fValue = listValues[t+1]
        if (bPrevious):
            # Previous flag is specified; skip those detections that occur at or later in time.
            if (fDateObs >= fDateObs_Current - 0.1):
                t += 2
                continue

        fDeltaTime = abs(fDateObs_Current - fDateObs)

        if (fDeltaTime < fMinDeltaTime):
            fMinDeltaTime = fDeltaTime
            fBestDateObs = fDateObs
            fBestValue = fValue
            bFound=True

        t += 2

    if (bFound):
        return (fBestDateObs, fBestValue)

    # Could not find a data point in the past -- use the nearest data point
    fMinDeltaTime = 1e9

    t = 0
    while (t<slen):
        fDateObs = listValues[t]
        fValue = listValues[t+1]
        fDeltaTime = abs(fDateObs_Current - fDateObs)

        if (fDeltaTime < fMinDeltaTime):
            fMinDeltaTime = fDeltaTime
            fBestDateObs = fDateObs
            fBestValue = fValue

        t += 2

    return (fBestDateObs, fBestValue)

# GetDetection
# Input:
#   listPixelsPerHourX: A list of DateObs:PixelsPerHourX pairs
#   listPixelsPerHourY: A list of DateObs:PixelsPerHourY pairs
#   listX: A list of DateObs:X pairs
#   listY: A list of DateObs:Y pairs
#   bPrevious: When set to TRUE, initially only looks for detections that occur prior to the given timestamp
#   fDateObs_Current: The timestamp associated with the current detection
# Output:
#   pX[0]: The DateObs associated with the closet matching detection
#   pixelsPerHourX[1]: The X-movement, in pixels/hour, of the closest matching detection
#   pixelsPerHourY[1]: The Y-movement, in pixels/hour, of the closest matching detection
#   pX[1]: The X coordinate of the closest matching detection
#   pY[1]: The Y coordinate of the closest matching detection
@njit
def GetDetection(listPixelsPerHourX, listPixelsPerHourY, listX, listY, bPrevious, fDateObs_Current):

    pixelsPerHourX = GetDetection2(listPixelsPerHourX, bPrevious, fDateObs_Current)
    pixelsPerHourY = GetDetection2(listPixelsPerHourY, bPrevious, fDateObs_Current)
    pX = GetDetection2(listX, bPrevious, fDateObs_Current)
    pY = GetDetection2(listY, bPrevious, fDateObs_Current)

    return (pX[0], pixelsPerHourX[1], pixelsPerHourY[1], pX[1], pY[1])

# AddDetectID
# Input:
#   listDetectIdentifiers: A list of detection identifiers
#   nDetectID: The identifier of the detection to be added
# Output:
#   Whether or not the detection identifier is already in the list
@njit
def AddDetectID(listDetectIdentifiers, nDetectID):

    slen = len(listDetectIdentifiers)
    for t in range(slen):
        detect_id = listDetectIdentifiers[t]
        if (detect_id==nDetectID):
            # Detection identifier already in list
            return False

    # Detection identifier not yet in list -- add it
    listDetectIdentifiers.append(types.int64(nDetectID))
    return True

# UpdateTrackDetectionIntervals
# Input:
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
#   fDateObs: The DateObs of a new detection
# Output:
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
@njit
def UpdateTrackDetectionIntervals(fDateObsFirstConfirmedDetection, fDateObsLastConfirmedDetection, fDateObs):

    if (fDateObsFirstConfirmedDetection < 0 or
            fDateObs < fDateObsFirstConfirmedDetection):
        fDateObsFirstConfirmedDetection = fDateObs

    if (fDateObsLastConfirmedDetection < 0 or
            fDateObs > fDateObsLastConfirmedDetection):
        fDateObsLastConfirmedDetection = fDateObs

    return (fDateObsFirstConfirmedDetection, fDateObsLastConfirmedDetection)

# TrackPresentOnImg
# Summary:
#   Identifies whether or not a track is present on a given image.
#   If so, the associated detection on that image is added to the track.
#   The input/output fields would normally be contained within an object,
#     however, Numba requires primitive data types in order to work correctly.
# Input:
#   source_img_idx: The index of the image from which the track was initially created
#   listPixelsPerHourX: A list of detection X-movements, in pixels/hour
#   listPixelsPerHourY: A list of detection Y-movements, in pixels/hour
#   listX: A list of detection X coordinates
#   listY: A list of detection Y coordinates
#   listDetectIdentifiers: A list of detection identifiers
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
#   nNumDetectsAt2: The current number of detections found at 2 pixels within expected position
#   nNumDetectsAt3: The current number of detections found at 3 pixels within expected position
#   nNumDetectsAt4: The current number of detections found at 4 pixels within expected position
#   nNumDetectsAt5: The current number of detections found at 5 pixels within expected position
#   nNumDetectsAt6: The current number of detections found at 6 pixels within expected position
#   nNumDetectsAt7: The current number of detections found at 7 pixels within expected position
#   nNumDetectsAt8: The current number of detections found at 8 pixels within expected position
#   listDetectCounts: The list of detection cluster sizes, in pixel counts
#   num_img: The number of images in the current sequence
#   idx_img: The index of the current image being evaluated
#   width: Image width, in pixels
#   height: Image height, in pixels
#   listDateObs: The list of DateObs associated with the current image sequence
#   listImgDetections: The list of detection images
#   update_counts: Whether or not to update the detection count statistics
# Output:
#   source_img_idx: The index of the image from which the track was initially created
#   listPixelsPerHourX: A list of detection X-movements, in pixels/hour
#   listPixelsPerHourY: A list of detection Y-movements, in pixels/hour
#   listX: A list of detection X coordinates
#   listY: A list of detection Y coordinates
#   listDetectIdentifiers: A list of detection identifiers
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
#   first_confirmed_idx_img: The index of the first confirmed detection image
#   last_confirmed_idx_img: The index of the last confirmed detection image
#   nNumDetectsAt2: The current number of detections found at 2 pixels within expected position
#   nNumDetectsAt3: The current number of detections found at 3 pixels within expected position
#   nNumDetectsAt4: The current number of detections found at 4 pixels within expected position
#   nNumDetectsAt5: The current number of detections found at 5 pixels within expected position
#   nNumDetectsAt6: The current number of detections found at 6 pixels within expected position
#   nNumDetectsAt7: The current number of detections found at 7 pixels within expected position
#   nNumDetectsAt8: The current number of detections found at 8 pixels within expected position
#   listDetectCounts: The list of detection cluster sizes, in pixel counts
@njit
def TrackPresentOnImg(source_img_idx,
                      listPixelsPerHourX,
                      listPixelsPerHourY,
                      listX,
                      listY,
                      listDetectIdentifiers,
                      fDateObsFirstConfirmedDetection,
                      fDateObsLastConfirmedDetection,
                      first_confirmed_idx_img,
                      last_confirmed_idx_img,
                      nNumDetectsAt2,
                      nNumDetectsAt3,
                      nNumDetectsAt4,
                      nNumDetectsAt5,
                      nNumDetectsAt6,
                      nNumDetectsAt7,
                      nNumDetectsAt8,
                      listDetectCounts,
                      num_img,
                      idx_img,
                      width,
                      height,
                      listDateObs,
                      listImgDetections,
                      update_counts):

    fDateObs_Current = listDateObs[idx_img]

    # Use the motion of the previous detection to predict where the current detection should be located
    res = GetDetection(listPixelsPerHourX, listPixelsPerHourY, listX, listY, True, fDateObs_Current)

    fDateObs_Adjacent = res[0]
    fPixelsPerHourX = res[1]
    fPixelsPerHourY = res[2]
    fX = res[3]
    fY = res[4]

    fElapsedTimeInSec = fDateObs_Current - fDateObs_Adjacent
    fElapsedTimeInHours = fElapsedTimeInSec / 3600.0

    fExpectedX = fX + (fElapsedTimeInHours * fPixelsPerHourX)
    fExpectedY = fY + (fElapsedTimeInHours * fPixelsPerHourY)

    i = int(fExpectedX + 0.5)
    j = int(fExpectedY + 0.5)

    fMaxDistance = 5.0 # 10.0
    r = 10
    len = (2 * r) + 1

    fClosestDistance = 1e9
    found_i = -1
    found_j = -1

    b=0
    while (b<len):

        a=0
        while (a<len):

            p = i - r + a
            q = j - r + b

            if (p<0 or p>=width or q<0 or q>=height):
                a=a+1
                continue

            v1 = listImgDetections[idx_img][q][p]

            if (v1 <= 0):
                a=a+1
                continue

            fDeltaX = p - i
            fDeltaY = q - j

            fDistance = sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY)

            if (fDistance > fMaxDistance):
                a=a+1
                continue

            if (fDistance < fClosestDistance):
                fClosestDistance = fDistance
                found_v1 = v1
                found_i = p
                found_j = q

            a=a+1

        b = b+1

    if (found_i < 0 or
        found_j < 0):
        return (source_img_idx,
                listPixelsPerHourX,
                listPixelsPerHourY,
                listX,
                listY,
                listDetectIdentifiers,
                fDateObsFirstConfirmedDetection,
                fDateObsLastConfirmedDetection,
                first_confirmed_idx_img,
                last_confirmed_idx_img,
                nNumDetectsAt2,
                nNumDetectsAt3,
                nNumDetectsAt4,
                nNumDetectsAt5,
                nNumDetectsAt6,
                nNumDetectsAt7,
                nNumDetectsAt8,
                listDetectCounts)

    if (update_counts):

        if (idx_img < first_confirmed_idx_img):
            first_confirmed_idx_img = idx_img

        if (idx_img > last_confirmed_idx_img):
            last_confirmed_idx_img = idx_img

        if (fClosestDistance < 2.0):
            nNumDetectsAt2 = nNumDetectsAt2 + 1
        elif (fClosestDistance < 3.0):
            nNumDetectsAt3 = nNumDetectsAt3 + 1
        elif (fClosestDistance < 4.0):
            nNumDetectsAt4 = nNumDetectsAt4 + 1
        elif (fClosestDistance < 5.0):
            nNumDetectsAt5 = nNumDetectsAt5 + 1
        elif (fClosestDistance < 6.0):
            nNumDetectsAt6 = nNumDetectsAt6 + 1
        elif (fClosestDistance < 7.0):
            nNumDetectsAt7 = nNumDetectsAt7 + 1
        elif (fClosestDistance < 8.0):
            nNumDetectsAt8 = nNumDetectsAt8 + 1
    else:

        nDetectID = (idx_img * width * height) + (found_i + width * found_j)
        bResult = AddDetectID(listDetectIdentifiers, nDetectID)

        if (bResult):
            # Detection not already in list -- incorporate the detection into the track velocity
            detect = [found_i, found_j]
            AddTrackVelocity(listPixelsPerHourX,
                             listPixelsPerHourY,
                             listX,
                             listY,
                             fDateObs_Current,
                             fElapsedTimeInSec,
                             found_i - fX,
                             found_j - fY,
                             detect)

        res = UpdateTrackDetectionIntervals(fDateObsFirstConfirmedDetection, fDateObsLastConfirmedDetection, fDateObs_Current)
        fDateObsFirstConfirmedDetection = res[0]
        fDateObsLastConfirmedDetection = res[1]

        res = UpdateTrackDetectionIntervals(fDateObsFirstConfirmedDetection, fDateObsLastConfirmedDetection, fDateObs_Adjacent)
        fDateObsFirstConfirmedDetection = res[0]
        fDateObsLastConfirmedDetection = res[1]

        listDetectCounts.append(types.int64(found_v1))

    return (source_img_idx,
            listPixelsPerHourX,
            listPixelsPerHourY,
            listX,
            listY,
            listDetectIdentifiers,
            fDateObsFirstConfirmedDetection,
            fDateObsLastConfirmedDetection,
            first_confirmed_idx_img,
            last_confirmed_idx_img,
            nNumDetectsAt2,
            nNumDetectsAt3,
            nNumDetectsAt4,
            nNumDetectsAt5,
            nNumDetectsAt6,
            nNumDetectsAt7,
            nNumDetectsAt8,
            listDetectCounts)

# CollectDetections
# Summary:
#   Collects detects from each image that can be associated with the current track
#   The input/output fields would normally be contained within an object,
#     however, Numba requires primitive data types in order to work correctly.
# Input:
#   source_img_idx: The index of the image from which the track was initially created
#   listPixelsPerHourX: A list of detection X-movements, in pixels/hour
#   listPixelsPerHourY: A list of detection Y-movements, in pixels/hour
#   listX: A list of detection X coordinates
#   listY: A list of detection Y coordinates
#   listDetectIdentifiers: A list of detection identifiers
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
#   first_confirmed_idx_img: The index of the first confirmed detection image
#   last_confirmed_idx_img: The index of the last confirmed detection image
#   nNumDetectsAt2: The current number of detections found at 2 pixels within expected position
#   nNumDetectsAt3: The current number of detections found at 3 pixels within expected position
#   nNumDetectsAt4: The current number of detections found at 4 pixels within expected position
#   nNumDetectsAt5: The current number of detections found at 5 pixels within expected position
#   nNumDetectsAt6: The current number of detections found at 6 pixels within expected position
#   nNumDetectsAt7: The current number of detections found at 7 pixels within expected position
#   nNumDetectsAt8: The current number of detections found at 8 pixels within expected position
#   listDetectCounts: A list of detection cluster sizes, in pixels
#   num_img: The number of images in the current sequence
#   width: Image width, in pixels
#   height: Image height, in pixels
#   listDateObs: List of DateObs associated with the current image sequence
#   listImgDetections: List of detection images
#   update_counts: Whether or not to update the detection count statistics
# Output:
#   source_img_idx: The index of the image from which the track was initially created
#   fDateObsFirstConfirmedDetection: The DateObs of the first confirmed detection
#   fDateObsLastConfirmedDetection: The DateObs of the last confirmed detection
#   first_confirmed_idx_img: The index of the first confirmed detection image
#   last_confirmed_idx_img: The index of the last confirmed detection image
#   nNumDetectsAt2: The current number of detections found at 2 pixels within expected position
#   nNumDetectsAt3: The current number of detections found at 3 pixels within expected position
#   nNumDetectsAt4: The current number of detections found at 4 pixels within expected position
#   nNumDetectsAt5: The current number of detections found at 5 pixels within expected position
#   nNumDetectsAt6: The current number of detections found at 6 pixels within expected position
#   nNumDetectsAt7: The current number of detections found at 7 pixels within expected position
#   nNumDetectsAt8: The current number of detections found at 8 pixels within expected position
@njit
def CollectDetections(source_img_idx,
                      listPixelsPerHourX,
                      listPixelsPerHourY,
                      listX,
                      listY,
                      listDetectIdentifiers,
                      fDateObsFirstConfirmedDetection,
                      fDateObsLastConfirmedDetection,
                      first_confirmed_idx_img,
                      last_confirmed_idx_img,
                      nNumDetectsAt2,
                      nNumDetectsAt3,
                      nNumDetectsAt4,
                      nNumDetectsAt5,
                      nNumDetectsAt6,
                      nNumDetectsAt7,
                      nNumDetectsAt8,
                      listDetectCounts,
                      num_img,
                      width,
                      height,
                      listDateObs,
                      listImgDetections,
                      update_counts):

    cur_idx = source_img_idx

    while (True):

        res = TrackPresentOnImg(source_img_idx,
                                listPixelsPerHourX,
                                listPixelsPerHourY,
                                listX,
                                listY,
                                listDetectIdentifiers,
                                fDateObsFirstConfirmedDetection,
                                fDateObsLastConfirmedDetection,
                                first_confirmed_idx_img,
                                last_confirmed_idx_img,
                                nNumDetectsAt2,
                                nNumDetectsAt3,
                                nNumDetectsAt4,
                                nNumDetectsAt5,
                                nNumDetectsAt6,
                                nNumDetectsAt7,
                                nNumDetectsAt8,
                                listDetectCounts,
                                num_img,
                                cur_idx,
                                width,
                                height,
                                listDateObs,
                                listImgDetections,
                                update_counts)

        source_img_idx = res[0]
        listPixelsPerHourX = res[1]
        listPixelsPerHourY = res[2]
        listX = res[3]
        listY = res[4]
        listDetectIdentifiers = res[5]
        fDateObsFirstConfirmedDetection = res[6]
        fDateObsLastConfirmedDetection = res[7]
        first_confirmed_idx_img = res[8]
        last_confirmed_idx_img = res[9]
        nNumDetectsAt2 = res[10]
        nNumDetectsAt3 = res[11]
        nNumDetectsAt4 = res[12]
        nNumDetectsAt5 = res[13]
        nNumDetectsAt6 = res[14]
        nNumDetectsAt7 = res[15]
        nNumDetectsAt8 = res[16]
        listDetectCounts = res[17]

        cur_idx = (cur_idx+1) % num_img

        if (cur_idx == source_img_idx):
            break

    return (source_img_idx,
            fDateObsFirstConfirmedDetection,
            fDateObsLastConfirmedDetection,
            first_confirmed_idx_img,
            last_confirmed_idx_img,
            nNumDetectsAt2,
            nNumDetectsAt3,
            nNumDetectsAt4,
            nNumDetectsAt5,
            nNumDetectsAt6,
            nNumDetectsAt7,
            nNumDetectsAt8)

# ComputeDetectionVelocityInOrder
# Summary:
#   Updates the detection velocities so that they are computed in order of image index
# Input:
#   listDetectionIdentifiers: A list of detection identifiers
#   listDateObs: The list of DateObs associated with the current image sequence
# Output:
#   listPixelsPerHourX: A list of DateObs:PixelsPerHourX pairs
#   listPixelsPerHourY: A list of DateObs:PixelsPerHourY pairs
#   listX: A list of DateObs:X pairs
#   listY: A list of DateObs:Y pairs
@njit
def ComputeDetectionVelocityInOrder(listDetectIdentifiers, listDateObs):

    listDetectIdentifiers.sort()

    listPixelsPerHourX = typed.List.empty_list(types.double)
    listPixelsPerHourY = typed.List.empty_list(types.double)
    listX = typed.List.empty_list(types.double)
    listY = typed.List.empty_list(types.double)

    slen = len(listDetectIdentifiers)

    prev_idx_img = 0
    prev_x = 0
    prev_y = 0

    for t in range(slen):

        nDetectID = listDetectIdentifiers[t]
        idx_img = int((nDetectID) / (1024*1024))
        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        if (0==t):
            prev_idx_img = idx_img
            prev_x = x
            prev_y = y
            continue

        fDateObsPrev = listDateObs[prev_idx_img]
        fPrevX = prev_x
        fPrevY = prev_y

        fDateObsCur = listDateObs[idx_img]
        fCurX = x
        fCurY = y

        fTimeElapsedInSec = fDateObsCur - fDateObsPrev
        fDeltaX = fCurX - fPrevX
        fDeltaY = fCurY - fPrevY

        detect = [fCurX, fCurY]
        AddTrackVelocity(listPixelsPerHourX,
                         listPixelsPerHourY,
                         listX,
                         listY,
                         fDateObsCur,
                         fTimeElapsedInSec,
                         fDeltaX,
                         fDeltaY,
                         detect)

        prev_idx_img = idx_img
        prev_x = x
        prev_y = y

    return (listPixelsPerHourX, listPixelsPerHourY, listX, listY)

# ComputeTrackMotionCoefficients
# Summary:
#   This function was used to compute a 1st or 2nd order fit for a given track
#   No longer used, since such fitting allowed more false tracks than the current
#     approach of enforcing proximity limits on a set of detections.
# Input:
#   track: A given track to evaluate
#   img_list: The list of images for the current image sequence
#   debug: A flag used for debugging purposes
# Modifies:
#   Track vector coefficients X and Y
def ComputeTrackMotionCoefficients(track, img_list, debug):

    vecTime = []
    vecX = []
    vecY = []

    slen = len(track.listDetectIdentifiers)
    for t in range(slen):

        nDetectID = track.listDetectIdentifiers[t]
        idx_img = int((nDetectID) / (1024*1024))
        fDateObs = img_list[idx_img].fDateObs
        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        vecTime.append(fDateObs)
        vecX.append(x)
        vecY.append(y)

    if (debug):
        slen = len(vecX)
        for t in range(slen):
            print("  ("+repr(vecTime[t])+", " +repr(vecX[t])+", "+repr(vecY[t])+")")
        return

    track.vecCoeffX.clear()
    track.vecCoeffY.clear()

    track.vecCoeffX = np.polyfit(vecTime, vecX, SOHO_FIT_ORDER)
    track.vecCoeffY = np.polyfit(vecTime, vecY, SOHO_FIT_ORDER)

# GetXYForTrackImgIndex
# Input:
#   track: A given track to be evaluated
#   idx_img: The index of the image on which to return the track position
# Output:
#   Found: Whether or not the track was found on the given image
#   x: The X coordinate of the track on the given image
#   y: The Y coordinate of the track on the given image
def GetXYForTrackImgIndex(track, idx_img):
    slen = len(track.listDetectIdentifiers)

    for t in range(slen):
        nDetectID = track.listDetectIdentifiers[t]
        nDetectImgIdx = int((nDetectID)/(1024*1024))

        if (nDetectImgIdx != idx_img):
            continue

        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        return (True, x, y)

    return (False, 0, 0)

# ComputeTrackPosition
# Input:
#   track: A given track to evaluate
#   idx_img: The image on which the track position is to be computed
#   fDateObs: The DateObs associated with the image on which the track position is to be computed
# Output:
#   X: The computed X position of the track
#   Y: The computed Y position of the track
def ComputeTrackPosition(track, idx_img, fDateObs):

    if (idx_img>=0):
        res = GetXYForTrackImgIndex(track, idx_img)
    else:
        res = (False, 0, 0)

    if (True == res[0]):
        # Return the XY of the found detection
        return (res[1], res[2])

    # Detection not found -- compute position
    res = GetDetection(track.listPixelsPerHourX,
                       track.listPixelsPerHourY,
                       track.listX,
                       track.listY,
                       False,
                       fDateObs)

    fDateObs_Source = res[0]
    fPixelsPerHourX = res[1]
    fPixelsPerHourY = res[2]
    fX = res[3]
    fY = res[4]

    fElapsedTimeInHours = fDateObs - fDateObs_Source
    fElapsedTimeInHours /= 3600.0

    fComputedX = fX + (fElapsedTimeInHours * fPixelsPerHourX)
    fComputedY = fY + (fElapsedTimeInHours * fPixelsPerHourY)

    return (fComputedX, fComputedY)

# ComputeTrackPositions
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Modifies:
#   track.vectorPositions: The track positions across the image sequence
def ComputeTrackPositions(track, num_img, img_list):

    xy = ComputeTrackPosition(track, track.first_confirmed_idx_img, track.fDateObsFirstConfirmedDetection)
    track.first_confirmed_x = int(xy[0] + 0.5)
    track.first_confirmed_y = int(xy[1] + 0.5)

    xy = ComputeTrackPosition(track, track.last_confirmed_idx_img, track.fDateObsLastConfirmedDetection)
    track.last_confirmed_x = int(xy[0] + 0.5)
    track.last_confirmed_y = int(xy[1] + 0.5)

    track.vectorPositions.clear()

    for t in range(num_img):
        xy = ComputeTrackPosition(track, t, img_list[t].fDateObs)
        track.vectorPositions.append((int(xy[0]+0.5),int(xy[1]+0.5)))

# ComputeDistanceToSun
# Input:
#   fX: A given X coordinate
#   fY: A given Y coordinate
# Output:
#   fDistance: The distance to the center of the image from the given X,Y location
def ComputeDistanceToSun(fX, fY):

    fDeltaX = fX - 512
    fDeltaY = fY - 512

    fDistance = sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY)

    return fDistance

# ComputeTrackSunMotionVector
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.fSunMotionVector: The motion vector of the track in the Sun direction.
def ComputeTrackSunMotionVector(track):

    # Compute the motion vector using the first and last confirmed detections
    fX1 = track.first_confirmed_x
    fY1 = track.first_confirmed_y
    fX2 = track.last_confirmed_x
    fY2 = track.last_confirmed_y

    fDeltaX = fX2 - fX1
    fDeltaY = fY2 - fY1
    fDistance = sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY)

    fSunDistance1 = ComputeDistanceToSun(fX1, fY1)
    fSunDistance2 = ComputeDistanceToSun(fX2, fY2)

    fDeltaSunDistance = fSunDistance2 - fSunDistance1

    if (abs(fDistance) > 0.01):
        track.fSunMotionVector = fDeltaSunDistance / fDistance
    else:
        track.fSunMotionVector = 0

# def GetAverageTrackVelocity(track, num_img, img_list):
#
#     fDateObs_First = img_list[0].fDateObs
#     fDateObs_Last = img_list[num_img - 1].fDateObs
#
#     res = GetDetection(track, False, fDateObs_First)
#     fDateObs_Found = res[0]
#     fPixelsPerHourX_1 = res[1]
#     fPixelsPerHourY_1 = res[2]
#     fX_1 = res[3]
#     fY_1 = res[4]
#
#     res = GetDetection(track, False, fDateObs_Last)
#     fDateObs_Found = res[0]
#     fPixelsPerHourX_2 = res[1]
#     fPixelsPerHourY_2 = res[2]
#     fX_2 = res[3]
#     fY_2 = res[4]
#
#     fPixelsPerHourX = 0.5 * (fPixelsPerHourX_1 + fPixelsPerHourX_2)
#     fPixelsPerHourY = 0.5 * (fPixelsPerHourY_1 + fPixelsPerHourY_2)
#
#     return (fPixelsPerHourX, fPixelsPerHourY)

# GetAverageTrackVelocity2
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Output:
#   fPixelsPerHourX: The X-motion of the track, in pixels/hour
#   fPixelsPerHourY: The Y-motion of the track, in pixels/hour
def GetAverageTrackVelocity2(track, num_img, img_list):

    fTimeElapsedInSec = track.fDateObsLastConfirmedDetection - track.fDateObsFirstConfirmedDetection

    if (abs(fTimeElapsedInSec) > 0.1):
        fTimeElapsedInHours = fTimeElapsedInSec / 3600.0
        fDeltaX = track.last_confirmed_x - track.first_confirmed_x
        fDeltaY = track.last_confirmed_y - track.first_confirmed_y

        fPixelsPerHourX = fDeltaX / (fTimeElapsedInHours)
        fPixelsPerHourY = fDeltaY / (fTimeElapsedInHours)
    else:
        fPixelsPerHourX = 0
        fPixelsPerHourY = 0

    return (fPixelsPerHourX, fPixelsPerHourY)

# ComputeTrackDirection
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Modifies:
#   track.fDirection: The computed direction of the track in the image
def ComputeTrackDirection(track, num_img, img_list):

    vel = GetAverageTrackVelocity2(track, num_img, img_list)
    track.fDirection = R2D * math.atan2(vel[1], vel[0])

# ComputeTrackGridSection
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.nGridSection: The location of the track in the image grid (integer in [0,15])
def ComputeTrackGridSection(track):

    # Compute the location of the object at midpoint
    fComputedX = 0.5 * (track.first_confirmed_x + track.last_confirmed_x)
    fComputedY = 0.5 * (track.first_confirmed_y + track.last_confirmed_y)

    fX = fComputedX / 1024.0
    fY = fComputedY / 1024.0

    nX = int(4.0 * fX + 0.5)
    nY = int(4.0 * fY + 0.5)


    if (nX < 0):
        nX = 0
    elif (nX >= 4):
        nX = 3

    if (nY < 0):
        nY = 0
    elif (nY >= 4):
        nY = 3

    track.nGridSection = nX + (4*nY)

# epoch_seconds_to_gregorian_date
# Summary:
#   Helper function to convert UNIX timestamp to year/month/day
#   Used to compute the month index of a given track
# Sourced from:
#   https://stackoverflow.com/questions/35796786/how-to-calculate-the-current-month-from-python-time-time
# Input:
#   eseconds: A UNIX timestamp (epoch of 1970-01-01 00:00:00)
# Output:
#   Y: Gregorian calendar year
#   M: Gregorian calendar month
#   D: Gregorian calendar day
def epoch_seconds_to_gregorian_date(eseconds):
    # Algorithm parameters for Gregorian calendar
    y = 4716; j = 1401; m = 2; n = 12; r = 4; p = 1461
    v = 3; u = 5; s = 153; w = 2; B = 274277; C = -38

    #Julian day, rounded
    J = int(0.5 + eseconds / 86400.0 + 2440587.5)

    f = J + j + (((4 * J + B) // 146097) * 3) // 4 + C
    e = r * f + v
    g = (e % p) // r
    h = u * g + w
    D = (h % s) // u + 1
    M = (h // s + m) % n + 1
    Y = (e // p) - y + (n + m - M) // n

    return Y, M, D

# ComputeTrackMonthIndex
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.nMonthIndex: The month index associated with the given track
def ComputeTrackMonthIndex(track):

    ymd = epoch_seconds_to_gregorian_date(track.fDateObsFirstConfirmedDetection)
    track.nMonthIndex = ymd[1] - 1

# ComputeTrackVelocity
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Modifies:
#   track.fVelocity: The average track velocity from first to last confirmed detection
def ComputeTrackVelocity(track, num_img, img_list):

    vel = GetAverageTrackVelocity2(track, num_img, img_list)
    fPixelsPerHourX = vel[0]
    fPixelsPerHourY = vel[1]

    track.fVelocity = sqrt(fPixelsPerHourX * fPixelsPerHourX + fPixelsPerHourY * fPixelsPerHourY)

# ComputeTrackAttributes
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Modifies:
#   Track sun motion vector
#   Track direction
#   Track grid section
#   Track velocity
#   Track month index
def ComputeTrackAttributes(track, num_img, img_list):

    ComputeTrackSunMotionVector(track)
    ComputeTrackDirection(track, num_img, img_list)
    ComputeTrackGridSection(track)
    ComputeTrackVelocity(track, num_img, img_list)
    ComputeTrackMonthIndex(track)

# ComputeR2
# Summary:
#   Was used as part of evaluating the quality of a track
#   No longer used since false tracks could still have high R2 fit
# Input:
#   vecTime: A vector of timestamps
#   vecCoord: A vector of coordinates
#   vecCoeff: A vector of coefficients for fitting
# Output:
#   The R2 value for the requested fit
def ComputeR2(vecTime, vecCoord, vecCoeff):

    fCount = 0
    fSum = 0

    slen = len(vecCoord)

    for t in range(slen):
        fSum += vecCoord[t]
        fCount += 1.0

    fAvg = fSum / fCount

    fSumSqResiduals = 0
    fSumSquares = 0

    for t in range(slen):
        fDelta = vecCoord[t] - fAvg
        fSumSquares += fDelta*fDelta

        fDateObs = vecTime[t]

        if (2==SOHO_FIT_ORDER):
            fComputedValue = vecCoeff[2] + vecCoeff[1] * fDateObs + (vecCoeff[0] * fDateObs * fDateObs)
        elif (1==SOHO_FIT_ORDER):
            fComputedValue = vecCoeff[1] + vecCoeff[0] * fDateObs

        fDelta = vecCoord[t] - fComputedValue
        fSumSqResiduals += fDelta*fDelta

    if (abs(fSumSquares) < 0.000000000001):
        return 0

    return 1.0 - (fSumSqResiduals / fSumSquares)

# ComputeTrackQuality_Fit
# Input:
#   track: A given track to evaluate
#   img_list: A list of images in the current sequence
# Output:
#   The best fit of the track in its X- and Y-motion
def ComputeTrackQuality_Fit(track, img_list):

    vecTime = []
    vecX = []
    vecY = []

    slen = len(track.listDetectIdentifiers)

    for t in range(slen):

        nDetectID = track.listDetectIdentifiers[t]
        idx_img = int(nDetectID / (1024*1024))
        fDateObs = img_list[idx_img].fDateObs
        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        vecTime.append(fDateObs)
        vecX.append(x)
        vecY.append(y)

    fR2_X = ComputeR2(vecTime, vecX, track.vecCoeffX)
    fR2_Y = ComputeR2(vecTime, vecY, track.vecCoeffY)

    track_r2 = min(fR2_X, fR2_Y)
    return track_r2

# ComputeFilteredSum
# Input:
#   listValues: A list of values to be evaluated
# Output:
#   A summation with the lowest and highest values omitted
def ComputeFilteredSum(listValues):

    listValues.sort()
    slen = len(listValues)
    idx_last = slen-1

    out = 0
    for t in range(slen):
        if (0==t):
            continue
        if (t==idx_last):
            break
        out += listValues[t]

    return out

# ComputeTrackQuality
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Modifies:
#   track.fGlobalQuality: The global quality score of the track
#   track.fQuality: The quality of the track without modifiers (no longer used)
def ComputeTrackQuality(track, num_img, img_list):

    fR2 = 0 # ComputeTrackQuality_Fit(track, img_list)
    track.fFit_R2 = fR2

    fNumImg = num_img

    fQuality = 4.0 * (track.nNumGT0)
    fQuality += 3.0 * (track.nNumGT1)
    fQuality += 2.0 * (track.nNumGT2)
    fQuality += 1.0 * (track.nNumGT3)
    fQuality /= 4.0 * fNumImg

    # fQuality = 7.0 * (track.nNumDetectsAt2)
    # fQuality += 6.0 * (track.nNumDetectsAt3)
    # fQuality += 5.0 * (track.nNumDetectsAt4)
    # fQuality += 4.0 * (track.nNumDetectsAt5)
    # fQuality += 3.0 * (track.nNumDetectsAt6)
    # fQuality += 2.0 * (track.nNumDetectsAt7)
    # fQuality += 1.0 * (track.nNumDetectsAt8)
    #
    # fQuality /= 7.0 * fNumImg
    # fQuality *= fR2

    track_quality.ComputeTrackQuality_SunMotionVector(track)
    track_quality.ComputeTrackQuality_Direction(track)
    track_quality.ComputeTrackQuality_GridSection(track)
    track_quality.ComputeTrackQuality_Velocity(track)
    track_quality.ComputeTrackQuality_GridDirection(track)

    # Compute global track quality

    nFilteredSum = ComputeFilteredSum(track.listDetectCounts)
    fFilteredSum = min(100, nFilteredSum)
    fFilteredSum /= 100.0
    fDetectCount = min(20, len(track.listDetectIdentifiers))
    fDetectCount /= 20.0
    fQualityR2 = fR2
    fQualityR2 -= 0.990
    fQualityR2 /= 0.01

    if (fQualityR2 < 0.0):
        fQualityR2 = 0.0
    elif (fQualityR2 > 1.0):
        fQualityR2 = 1.0

    track.fGlobalQuality = 0.80 * fQuality + 0.10 * fFilteredSum + 0.10 * fDetectCount

    if (track.bFlaggedSunMotionVector):
        track.fGlobalQuality *= 0.70 # 0.50

    if (track.bFlaggedDirection):
        track.fGlobalQuality *= 0.70 # 0.50

    if (track.bFlaggedGridSection):
        track.fGlobalQuality *= 0.80

    if (track.bFlaggedVelocity):
        track.fGlobalQuality *= 0.75

    if (track.bFlaggedGridDirection):
        track.fGlobalQuality *= 0.75

    if (track.nNumGT5 > 0):
        track.fGlobalQuality *= 0.50

    if (abs(track.median_delta_y) < 2):
        track.fGlobalQuality *= 0.50

    track.fQuality = fQuality

# IsEdgeTrack
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
# Output:
#   Whether or not the track moves along the edges of the images
def IsEdgeTrack(track, num_img, img_list):

    width=1024
    height=1024
    vel = GetAverageTrackVelocity2(track, num_img, img_list)
    fPixelsPerHourX = vel[0]
    fPixelsPerHourY = vel[1]

    # Compute how far the track moves in four hours
    fTimeElapsedInHours = 4.0

    fDeltaX = fTimeElapsedInHours * (fPixelsPerHourX)
    fDeltaY = fTimeElapsedInHours * (fPixelsPerHourY)

    fMinMovement = 4.0

    if (track.source_img_x >= -10 and
            track.source_img_x <= 10 and
            abs(fDeltaX) < fMinMovement):
        return True

    if (track.source_img_x >= width-10 and
            track.source_img_x <= width + 10 and
            abs(fDeltaX) < fMinMovement):
        return True

    if (track.source_img_y >= -10 and
            track.source_img_y <= 10 and
            abs(fDeltaY) < fMinMovement):
        return True

    if (track.source_img_y >= height - 10 and
            track.source_img_y <= height + 10 and
            abs(fDeltaY) < fMinMovement):
        return True

    return False

# ValidateTrackDetections
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
#   img_list: The list of images in the current sequence
#   width: The image width, in pixels
#   height: The image height, in pixels
# Modifies:
#   track.median_delta_x
#   track.median_delta_y
#   track.median_x
#   track.median_y
#   track.nNumGT0
#   track.nNumGT1
#   track.nNumGT2
#   track.nNumGT3
#   track.nNumGT5
#   track.nNumGT7
def ValidateTrackDetections(track, num_img, img_list, width, height):

    track.nNumGT0 = 0
    track.nNumGT1 = 0
    track.nNumGT2 = 0
    track.nNumGT3 = 0
    track.nNumGT5 = 0
    track.nNumGT7 = 0

    listDetects = []

    slen = len(track.listDetectIdentifiers)

    for t in range(slen):

        nDetectID = track.listDetectIdentifiers[t]
        idx_img = int(nDetectID / int(1024*1024))
        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        new_detect = MyDetect()
        new_detect.idx_img = idx_img
        new_detect.x = x
        new_detect.y = y
        listDetects.append(new_detect)

    listDetects.sort(key=lambda x: x.idx_img, reverse=False)

    track.listDetectIdentifiers.clear()

    slen = len(listDetects)

    for t in range(slen):
        cur_detect = listDetects[t]
        idx_img = cur_detect.idx_img
        x = cur_detect.x
        y = cur_detect.y

        nDetectID = (idx_img * width * height) + (x + width * y)
        track.listDetectIdentifiers.append(nDetectID)

    listX=[]
    listY=[]
    listDeltaX=[]
    listDeltaY=[]

    fPixelsPerHourX=0
    fPixelsPerHourY=0
    fExpectedX=0
    fExpectedY=0
    bHasValidSpeed = False
    fPrevDateObs = 0
    prev_x = 0
    prev_y = 0

    slen = len(track.listDetectIdentifiers)

    for t in range(slen):

        nDetectID = track.listDetectIdentifiers[t]
        idx_img = int(nDetectID / int(1024*1024))
        fDateObs = img_list[idx_img].fDateObs
        x = nDetectID % 1024
        y = int(nDetectID / 1024) % 1024

        listX.append(x)
        listY.append(y)

        if (0 == t):
            fPrevDateObs = fDateObs
            prev_x = x
            prev_y = y
            continue

        listDeltaX.append((x - prev_x))
        listDeltaY.append((y - prev_y))

        fElapsedTimeInSec = fDateObs - fPrevDateObs

        if (abs(fElapsedTimeInSec)>0.1):
            fElapsedTimeInHours = fElapsedTimeInSec / 3600.0
        else:
            fElapsedTimeInHours = 0.1 / 3600.0

        if (bHasValidSpeed):
            fExpectedX = prev_x + fPixelsPerHourX * fElapsedTimeInHours
            fExpectedY = prev_y + fPixelsPerHourY * fElapsedTimeInHours

            fDeltaX = x - fExpectedX
            fDeltaY = y - fExpectedY
            fDistance = sqrt(fDeltaX * fDeltaX + fDeltaY * fDeltaY)

            if (fDistance > 7.0):
                track.nNumGT7 += 1
            elif (fDistance > 5.0):
                track.nNumGT5 += 1
            elif (fDistance > 3.0):
                track.nNumGT3 += 1
            elif (fDistance > 2.0):
                track.nNumGT2 += 1
            elif (fDistance > 1.0):
                track.nNumGT1 += 1
            elif (fDistance > 0.0):
                track.nNumGT0 += 1

        fDeltaX = x - prev_x
        fDeltaY = y - prev_y

        fPixelsPerHourX = fDeltaX / fElapsedTimeInHours
        fPixelsPerHourY = fDeltaY / fElapsedTimeInHours
        bHasValidSpeed = True

        fPrevDateObs = fDateObs
        prev_x = x
        prev_y = y

    track.median_delta_x = np.median(listDeltaX)
    track.median_delta_y = np.median(listDeltaY)
    track.median_x = np.median(listX)
    track.median_y = np.median(listY)

# ShouldKeepTrack
# Input:
#   track: A given track to be evaluated
#   img_list: The list of images in the current sequence
#   listDateObs: The list of DateObs associated with the images
#   width: Image width, in pixels
#   height: Image height, in pixels
# Output:
#   Whether or not the given track should be retained
def ShouldKeepTrack(track, img_list, listDateObs, listImgDetections, width, height):

    num_img = len(img_list)

    # First pass -- identify detections along the track and update track velocity accordingly
    res = CollectDetections(track.source_img_idx,
                                track.listPixelsPerHourX,
                                track.listPixelsPerHourY,
                                track.listX,
                                track.listY,
                                track.listDetectIdentifiers,
                                track.fDateObsFirstConfirmedDetection,
                                track.fDateObsLastConfirmedDetection,
                                track.first_confirmed_idx_img,
                                track.last_confirmed_idx_img,
                                track.nNumDetectsAt2,
                                track.nNumDetectsAt3,
                                track.nNumDetectsAt4,
                                track.nNumDetectsAt5,
                                track.nNumDetectsAt6,
                                track.nNumDetectsAt7,
                                track.nNumDetectsAt8,
                                track.listDetectCounts, num_img, width, height, listDateObs, listImgDetections, False)

    track.source_img_idx = res[0]
    track.fDateObsFirstConfirmedDetection = res[1]
    track.fDateObsLastConfirmedDetection = res[2]
    track.first_confirmed_idx_img = res[3]
    track.last_confirmed_idx_img = res[4]
    track.nNumDetectsAt2 = res[5]
    track.nNumDetectsAt3 = res[6]
    track.nNumDetectsAt4 = res[7]
    track.nNumDetectsAt5 = res[8]
    track.nNumDetectsAt6 = res[9]
    track.nNumDetectsAt7 = res[10]
    track.nNumDetectsAt8 = res[11]

    if (len(track.listDetectIdentifiers) < 4):
        return False

    res = ComputeDetectionVelocityInOrder(track.listDetectIdentifiers, listDateObs)

    track.listPixelsPerHourX = res[0]
    track.listPixelsPerHourY = res[1]
    track.listX = res[2]
    track.listY = res[3]

    # Second pass -- run once more with the updated track velocity
    # ComputeTrackMotionCoefficients(track, img_list, False)
    res = CollectDetections(track.source_img_idx,
                                track.listPixelsPerHourX,
                                track.listPixelsPerHourY,
                                track.listX,
                                track.listY,
                                track.listDetectIdentifiers,
                                track.fDateObsFirstConfirmedDetection,
                                track.fDateObsLastConfirmedDetection,
                                track.first_confirmed_idx_img,
                                track.last_confirmed_idx_img,
                                track.nNumDetectsAt2,
                                track.nNumDetectsAt3,
                                track.nNumDetectsAt4,
                                track.nNumDetectsAt5,
                                track.nNumDetectsAt6,
                                track.nNumDetectsAt7,
                                track.nNumDetectsAt8,
                                track.listDetectCounts, num_img, width, height, listDateObs, listImgDetections, True)

    track.source_img_idx = res[0]
    track.fDateObsFirstConfirmedDetection = res[1]
    track.fDateObsLastConfirmedDetection = res[2]
    track.first_confirmed_idx_img = res[3]
    track.last_confirmed_idx_img = res[4]
    track.nNumDetectsAt2 = res[5]
    track.nNumDetectsAt3 = res[6]
    track.nNumDetectsAt4 = res[7]
    track.nNumDetectsAt5 = res[8]
    track.nNumDetectsAt6 = res[9]
    track.nNumDetectsAt7 = res[10]
    track.nNumDetectsAt8 = res[11]

    if (num_img <= 6):
        # Must have at least three detections at 2 or better
        # And one detection at 3 or better
        if (track.nNumDetectsAt2 + track.nNumDetectsAt3 < 4 or
            track.nNumDetectsAt2 < 3):
            return False
    else:
        # Must have at least three detections at 2 or better
        # And two detections at 3 or better
        if (track.nNumDetectsAt2 + track.nNumDetectsAt3 < 5 or
            track.nNumDetectsAt2 < 3):
            return False

    ValidateTrackDetections(track, num_img, img_list, width, height)

    if (track.nNumGT7 > 0):
        return False

    # Compute the track positions
    ComputeTrackPositions(track, num_img, img_list)

    # Compute track attributes
    ComputeTrackAttributes(track, num_img, img_list)

    # Compute track quality
    ComputeTrackQuality(track, num_img, img_list)

    #if (track.fFit_R2 < SOHO_FIT_ORDER_CUTOFF):
    #    return False

    if (IsEdgeTrack(track, num_img, img_list)):
        return False

    return True

# ReduceTracks
# Input:
#   listTracks: A list of tracks to be evaluated
#   img_list: The list of images in the current sequence
#   listDateObs: The list of DateObs associated with the images
#   listImgDetections: The list of detection images
#   width: Image width, in pixels
#   height: Image height, in pixels
# Output:
#   A new list of tracks, after having performed track reduction.
def ReduceTracks(listTracks, img_list, listDateObs, listImgDetections, width, height):

    listTracks[:] = [trk for trk in listTracks if ShouldKeepTrack(trk, img_list, listDateObs, listImgDetections, width, height)]

    return listTracks

# PrintTracks
# Input:
#   listTracks: A list of track to be printed
#   img_list: The list of images for the current sequence
# Output:
#   Prints track information to stdout
def PrintTracks(listTracks, img_list):

    slen = len(listTracks)

    for t in range(slen):
        track = listTracks[t]
        print(repr(t+1)+". ("+repr(track.source_img_x)+", "+repr(track.source_img_y)+")" + " Q="+repr(track.fGlobalQuality) + " R2="+repr(track.fFit_R2))

        num_pos = len(track.vectorPositions)
        for q in range(num_pos):
            xy = track.vectorPositions[q]
            print("    ("+repr(xy[0])+", "+repr(xy[1])+")")

        if (0==t):
            ComputeTrackMotionCoefficients(track, img_list, True)

# CullToTopNTracks
# Input:
#   list_tracks: A list of tracks
#   N: The number of tracks to be retained
# Output:
#   A new list of tracks, preserving only the top N
def CullToTopNTracks(list_tracks, N):
    newlist = sorted(list_tracks, key=lambda x: x.fGlobalQuality, reverse=True)
    return newlist[:N]

# ConsolidateTracks2
# Input:
#   track1: First track
#   track2: Second track
#   num_img: The number of images in the current sequence
# Output:
#   Whether or not the two tracks are considered equivalent
def ConsolidateTracks2(track1, track2, num_img):

    nNumSameDetectIdentifiers = 0

    slen1 = len(track1.listDetectIdentifiers)
    slen2 = len(track2.listDetectIdentifiers)

    for t in range(slen1):

        nDetectID1 = track1.listDetectIdentifiers[t]

        for z in range(slen2):

            nDetectID2 = track2.listDetectIdentifiers[z]

            if (nDetectID1 == nDetectID2):
                nNumSameDetectIdentifiers = nNumSameDetectIdentifiers + 1

    nNumSameDetects = max(3, min(5, int(SOHO_PCT_SAME_DETECTS * num_img)))

    if (nNumSameDetectIdentifiers < nNumSameDetects):
        # Not the same track
        return False

    # Tracks 1 and 2 are considered the same

    # Mark track 2 for deletion
    track2.bMarkedForDeletion = True

    # Incorporate the motion of track 2 into track 1
    for t in range(num_img):
        xy1 = track1.vectorPositions[t]
        xy2 = track2.vectorPositions[t]

        x1 = xy1[0]
        y1 = xy1[1]

        x2 = xy2[0]
        y2 = xy2[1]

        x3 = x1 + x2
        y3 = y1 + y2

        xy3 = []
        xy3.append(x3)
        xy3.append(y3)

        track1.vectorPositions[t] = xy3

    # Update the number of combined tracks
    track1.nNumCombinedTracks = track1.nNumCombinedTracks + 1

# ConsolidateTracks
# Input:
#   list_tracks: A list of tracks to be consolidated
#   num_img: The number of images in the current sequence
# Output:
#   A new list of tracks, after performing consolidation.
def ConsolidateTracks(list_tracks, num_img):

    slen = len(list_tracks)

    for t in range(slen):

        track1 = list_tracks[t]

        if (track1.bMarkedForDeletion):
            continue

        z = t+1

        while (z<slen):

            track2 = list_tracks[z]

            ConsolidateTracks2(track1, track2, num_img)

            z = z+1

    # Delete the tracks that were marked for deletion
    list_tracks[:] = [trk for trk in list_tracks if not trk.bMarkedForDeletion]

    return list_tracks

# FinalizeMotion2
# Input:
#   track: A given track to be evaluated
#   num_img: The number of images in the current sequence
# Modifies:
#   track.vectorPositions: Divides position values by the number of combined tracks
def FinalizeMotion2(track, num_img):

    fSum = track.nNumCombinedTracks

    for t in range(num_img):
        xy = track.vectorPositions[t]

        x = xy[0]
        y = xy[1]

        x /= fSum
        x = int(x+0.5)

        y /= fSum
        y = int(y+0.5)

        xy2 = [x,y]
        track.vectorPositions[t] = xy2

# FinalizeMotion
# Input:
#   listTracks: A list of tracks
#   num_img: The number of images in the current sequence
# Modifies:
#   The vectorPosition member of each track
def FinalizeMotion(listTracks, num_img):

    slen = len(listTracks)
    for t in range(slen):
        FinalizeMotion2(listTracks[t], num_img)

