"""
File: track_quality.py
Note: This code computes track quality from various attributes (speed, direction, location)
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

warnings.filterwarnings("ignore")

# Class definitions
class MyTrack:
    pass

# Model of track directions for each grid section
m_listGridDirections = { 0:[], 1:[], 2:[], 3:[],
                         4:[], 5:[], 6:[], 7:[],
                         8:[], 9:[], 10:[], 11:[],
                         12:[], 13:[], 14:[], 15:[] }

# PopulateGridDirection
# Input:
#   None
# Modifies:
#   m_listGridDirections: List of likely track directions for each grid section
#   Track directions are encoded as a simple range of directions, in degrees
def PopulateGridDirection():

    m_listGridDirections[0].append([-55, -34])
    m_listGridDirections[0].append([30, 50])
    m_listGridDirections[0].append([64, 90])

    m_listGridDirections[1].append([30, 120])

    m_listGridDirections[2].append([42, 82])
    m_listGridDirections[2].append([100, 135])

    m_listGridDirections[3].append([-154, -123])
    m_listGridDirections[3].append([67, 150])

    m_listGridDirections[4].append([23, 41])
    m_listGridDirections[4].append([47, 78])

    m_listGridDirections[5].append([-180, 180])

    m_listGridDirections[6].append([-180, 180])

    m_listGridDirections[7].append([-146, -115])
    m_listGridDirections[7].append([75, 156])

    m_listGridDirections[8].append([-180, 180])

    m_listGridDirections[9].append([-180, 180])

    m_listGridDirections[10].append([-180, 180])

    m_listGridDirections[11].append([-180, 180])

    m_listGridDirections[12].append([-86, -28])

    m_listGridDirections[13].append([-110, -70])
    m_listGridDirections[13].append([-56, -30])
    m_listGridDirections[13].append([30, 53])
    m_listGridDirections[13].append([76, 108])

    m_listGridDirections[14].append([-130, -98])
    m_listGridDirections[14].append([-82, -48])

    m_listGridDirections[15].append([-153, -62])
    m_listGridDirections[15].append([110, 144])
    m_listGridDirections[15].append([165, 177])

# ComputeTrackQuality_SunMotionVector_Jan
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of January
def ComputeTrackQuality_SunMotionVector_Jan(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.80 and
            track.fSunMotionVector <= -0.20):
        bDesired = False

    if (track.fSunMotionVector >= 0.70):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Feb
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of February
def ComputeTrackQuality_SunMotionVector_Feb(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.70 and
            track.fSunMotionVector <= -0.25):
        bDesired = False

    if (track.fSunMotionVector >= 0.20):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Mar
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of March
def ComputeTrackQuality_SunMotionVector_Mar(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.85 and
            track.fSunMotionVector <= -0.12):
        bDesired = False

    if (track.fSunMotionVector >= 0.00):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Apr
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of April
def ComputeTrackQuality_SunMotionVector_Apr(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.80 and
            track.fSunMotionVector <= 0.10):
        bDesired = False

    if (track.fSunMotionVector >= 0.40):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_May
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of May
def ComputeTrackQuality_SunMotionVector_May(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.72 and
            track.fSunMotionVector <= 0.73):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Jun
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of June
def ComputeTrackQuality_SunMotionVector_Jun(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector <= -0.92):
        bDesired = False

    if (track.fSunMotionVector >= -0.72):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Jul
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of July
def ComputeTrackQuality_SunMotionVector_Jul(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.80 and
            track.fSunMotionVector <= 0.00):
        bDesired = False

    if (track.fSunMotionVector >= 0.50):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Aug
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of August
def ComputeTrackQuality_SunMotionVector_Aug(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= 0.10 and
            track.fSunMotionVector <= 0.70):
        bDesired = False

    if (track.fSunMotionVector >= 0.90):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Sep
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of September
def ComputeTrackQuality_SunMotionVector_Sep(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.80 and
            track.fSunMotionVector <= -0.20):
        bDesired = False

    if (track.fSunMotionVector >= 0.10):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Oct
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of October
def ComputeTrackQuality_SunMotionVector_Oct(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.80 and
            track.fSunMotionVector <= 0.05):
        bDesired = False

    if (track.fSunMotionVector >= 0.30):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Nov
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of November
def ComputeTrackQuality_SunMotionVector_Nov(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.72):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_Dec
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for the month of December
def ComputeTrackQuality_SunMotionVector_Dec(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector <= -0.93):
        bDesired = False

    if (track.fSunMotionVector >= -0.72):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector_All
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely sun motion vector for any month
def ComputeTrackQuality_SunMotionVector_All(track):

    # By default
    bDesired = True

    if (track.fSunMotionVector >= -0.72):
        bDesired = False

    return bDesired

# ComputeTrackQuality_SunMotionVector
# Input:
#   track: A given track to evaluate
# Modifies:
#   track.bFlaggedSunMotionVector
def ComputeTrackQuality_SunMotionVector(track):

    # By default
    bHasDesiredSunMotionVector = False

    idx = track.nMonthIndex

    if (0 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Jan(track)
    elif (1 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Feb(track)
    elif (2 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Mar(track)
    elif (3 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Apr(track)
    elif (4 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_May(track)
    elif (5 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Jun(track)
    elif (6 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Jul(track)
    elif (7 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Aug(track)
    elif (8 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Sep(track)
    elif (9 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Oct(track)
    elif (10 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Nov(track)
    elif (11 == idx):
        bHasDesiredSunMotionVector = ComputeTrackQuality_SunMotionVector_Dec(track)

    if (bHasDesiredSunMotionVector):
        return

    track.bFlaggedSunMotionVector = True

# ComputeTrackQuality_Direction_Jan
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of January
def ComputeTrackQuality_Direction_Jan(track):

    # By default
    bDesired = True

    if (track.fDirection >= -110.0 and
            track.fDirection <= 10.0):
        bDesired = False

    if (track.fDirection >= 110.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Feb
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of February
def ComputeTrackQuality_Direction_Feb(track):

    # By default
    bDesired = True

    if (track.fDirection >= -135.0 and
            track.fDirection <= -10.0):
        bDesired = False

    if (track.fDirection >= 55.0 and
            track.fDirection <= 140.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Mar
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of March
def ComputeTrackQuality_Direction_Mar(track):

    # By default
    bDesired = True

    if (track.fDirection <= -150.0):
        bDesired = False

    if (track.fDirection >= -110.0 and
            track.fDirection <= 20.0):
        bDesired = False

    if (track.fDirection >= 55.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Apr
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of April
def ComputeTrackQuality_Direction_Apr(track):

    # By default
    bDesired = True

    if (track.fDirection <= -150.0):
        bDesired = False

    if (track.fDirection >= -110.0 and
            track.fDirection <= 20.0):
        bDesired = False

    if (track.fDirection >= 80.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_May
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of May
def ComputeTrackQuality_Direction_May(track):

    # By default
    bDesired = True

    if (track.fDirection <= -140.0):
        bDesired = False

    if (track.fDirection >= -95.0 and
            track.fDirection <= 40.0):
        bDesired = False

    if (track.fDirection >= 110.0 and
            track.fDirection <= 165.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Jun
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of June
def ComputeTrackQuality_Direction_Jun(track):

    # By default
    bDesired = True

    if (track.fDirection <= -110.0):
        bDesired = False

    if (track.fDirection >= -55.0 and
            track.fDirection <= 65.0):
        bDesired = False

    if (track.fDirection >= 130.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Jul
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of July
def ComputeTrackQuality_Direction_Jul(track):

    # By default
    bDesired = True

    if (track.fDirection <= -80.0):
        bDesired = False

    if (track.fDirection >= -30.0 and
            track.fDirection <= 105.0):
        bDesired = False

    if (track.fDirection >= 143.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Aug
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of August
def ComputeTrackQuality_Direction_Aug(track):

    # By default
    bDesired = True

    if (track.fDirection <= -140):
        bDesired = False

    if (track.fDirection >= -110.0 and
            track.fDirection <= -45.0):
        bDesired = False

    if (track.fDirection >= -10.0 and
            track.fDirection <= 95.0):
        bDesired = False

    if (track.fDirection >= 175.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Sep
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of September
def ComputeTrackQuality_Direction_Sep(track):

    # By default
    bDesired = True

    if (track.fDirection <= -50.0):
        bDesired = False

    if (track.fDirection >= -30.0 and
            track.fDirection <= 127.0):
        bDesired = False

    if (track.fDirection >= 150.0 and
            track.fDirection <= 168.0):
        bDesired = False

    if (track.fDirection >= 179.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Oct
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of October
def ComputeTrackQuality_Direction_Oct(track):

    # By default
    bDesired = True

    if (track.fDirection <= -70.0):
        bDesired = False

    if (track.fDirection >= -30.0 and
            track.fDirection <= 116.0):
        bDesired = False

    if (track.fDirection >= 145.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Nov
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of November
def ComputeTrackQuality_Direction_Nov(track):

    # By default
    bDesired = True

    if (track.fDirection <= -100.0):
        bDesired = False

    if (track.fDirection >= -45.0 and
            track.fDirection <= 87.0):
        bDesired = False

    if (track.fDirection >= 138.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_Dec
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for the month of December
def ComputeTrackQuality_Direction_Dec(track):

    # By default
    bDesired = True

    if (track.fDirection <= -130.0):
        bDesired = False

    if (track.fDirection >= -75.0 and
            track.fDirection <= 47.0):
        bDesired = False

    if (track.fDirection >= 110.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction_All
# Input:
#   track: A given track to evaluate
# Output:
#   Whether or not the track has a statistically likely direction for any month
def ComputeTrackQuality_Direction_All(track):

    # By default
    bDesired = True

    if (track.fDirection <= -155.0):
        bDesired = False

    if (track.fDirection >= -35.0 and
            track.fDirection <= 24.0):
        bDesired = False

    if (track.fDirection >= 144.0):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Direction
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.bFlaggedDirection
def ComputeTrackQuality_Direction(track):

    # By default
    bHasDesiredDirection = False

    idx = track.nMonthIndex

    if (0 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Jan(track)
    elif (1 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Feb(track)
    elif (2 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Mar(track)
    elif (3 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Apr(track)
    elif (4 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_May(track)
    elif (5 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Jun(track)
    elif (6 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Jul(track)
    elif (7 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Aug(track)
    elif (8 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Sep(track)
    elif (9 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Oct(track)
    elif (10 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Nov(track)
    elif (11 == idx):
        bHasDesiredDirection = ComputeTrackQuality_Direction_Dec(track)

    if (bHasDesiredDirection):
        return

    track.bFlaggedDirection = True

# ComputeTrackQuality_GridSection_Jan
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of January
def ComputeTrackQuality_GridSection_Jan(track):

    # By default
    bDesired = False

    if (3==track.nGridSection or
            4==track.nGridSection or
            6==track.nGridSection or
            7==track.nGridSection or
            8==track.nGridSection or
            11==track.nGridSection or
            13==track.nGridSection or
            14==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Feb
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of February
def ComputeTrackQuality_GridSection_Feb(track):

    # By default
    bDesired = False

    if (0==track.nGridSection or
            1==track.nGridSection or
            3==track.nGridSection or
            4==track.nGridSection or
            5==track.nGridSection or
            7==track.nGridSection or
            9==track.nGridSection or
            10==track.nGridSection or
            11==track.nGridSection or
            12==track.nGridSection or
            13==track.nGridSection or
            14==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Mar
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of March
def ComputeTrackQuality_GridSection_Mar(track):

    # By default
    bDesired = False

    if (0==track.nGridSection or
            1==track.nGridSection or
            3==track.nGridSection or
            4==track.nGridSection or
            12==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Apr
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of April
def ComputeTrackQuality_GridSection_Apr(track):

    # By default
    bDesired = False

    if (1==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_May
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of May
def ComputeTrackQuality_GridSection_May(track):

    # By default
    bDesired = False

    if (1==track.nGridSection or
            2==track.nGridSection or
            3==track.nGridSection or
            13==track.nGridSection or
            14==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Jun
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of June
def ComputeTrackQuality_GridSection_Jun(track):

    # By default
    bDesired = False

    if (3==track.nGridSection or
            7==track.nGridSection or
            12==track.nGridSection or
            13==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Jul
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of July
def ComputeTrackQuality_GridSection_Jul(track):

    # By default
    bDesired = False

    if (3==track.nGridSection or
            7==track.nGridSection or
            12==track.nGridSection or
            14==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Aug
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of August
def ComputeTrackQuality_GridSection_Aug(track):

    # By default
    bDesired = False

    if (0==track.nGridSection or
            4==track.nGridSection or
            7==track.nGridSection or
            11==track.nGridSection or
            12==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Sep
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of September
def ComputeTrackQuality_GridSection_Sep(track):

    # By default
    bDesired = False

    if (3==track.nGridSection or
            7==track.nGridSection or
            12==track.nGridSection or
            13==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Oct
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of October
def ComputeTrackQuality_GridSection_Oct(track):

    # By default
    bDesired = False

    if (3==track.nGridSection or
            7==track.nGridSection or
            12==track.nGridSection or
            13==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Nov
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of November
def ComputeTrackQuality_GridSection_Nov(track):

    # By default
    bDesired = False

    if (1==track.nGridSection or
            2==track.nGridSection or
            3==track.nGridSection or
            13==track.nGridSection or
            14==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_Dec
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for the month of December
def ComputeTrackQuality_GridSection_Dec(track):

    # By default
    bDesired = False

    if (1==track.nGridSection or
            4==track.nGridSection or
            15==track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection_All
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the track has a statistically likely grid section for any month
def ComputeTrackQuality_GridSection_All(track):

    # By default
    bDesired = False

    if (1 == track.nGridSection or
            2 == track.nGridSection or
            3 == track.nGridSection or
            7 == track.nGridSection or
            12 == track.nGridSection or
            13 == track.nGridSection or
            14 == track.nGridSection or
            15 == track.nGridSection):
        bDesired=True
    return bDesired

# ComputeTrackQuality_GridSection
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.bFlaggedGridSection
def ComputeTrackQuality_GridSection(track):

    # By default
    bHasDesiredGridSection = False

    idx = track.nMonthIndex

    if (0 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Jan(track)
    elif (1 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Feb(track)
    elif (2 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Mar(track)
    elif (3 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Apr(track)
    elif (4 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_May(track)
    elif (5 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Jun(track)
    elif (6 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Jul(track)
    elif (7 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Aug(track)
    elif (8 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Sep(track)
    elif (9 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Oct(track)
    elif (10 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Nov(track)
    elif (11 == idx):
        bHasDesiredGridSection = ComputeTrackQuality_GridSection_Dec(track)

    if (bHasDesiredGridSection):
        return

    track.bFlaggedGridSection = True

# ComputeTrackQuality_Velocity_Jan
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of January
def ComputeTrackQuality_Velocity_Jan(track):

    bDesired = True

    if (track.fVelocity < 34.7 or track.fVelocity > 97.7):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Feb
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of February
def ComputeTrackQuality_Velocity_Feb(track):

    bDesired = True

    if (track.fVelocity < 34.7 or track.fVelocity > 97.7):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Mar
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of March
def ComputeTrackQuality_Velocity_Mar(track):

    bDesired = True

    if (track.fVelocity < 68.9 or track.fVelocity > 97.7):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Apr
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of April
def ComputeTrackQuality_Velocity_Apr(track):

    bDesired = True

    if (track.fVelocity < 45 or track.fVelocity > 83):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_May
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of May
def ComputeTrackQuality_Velocity_May(track):

    bDesired = True

    if (track.fVelocity < 37.6 or track.fVelocity > 65.1):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Jun
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of June
def ComputeTrackQuality_Velocity_Jun(track):

    bDesired = True

    if (track.fVelocity < 33 or track.fVelocity > 52):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Jul
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of July
def ComputeTrackQuality_Velocity_Jul(track):

    bDesired = True

    if (track.fVelocity < 40 or track.fVelocity > 90):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Aug
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of August
def ComputeTrackQuality_Velocity_Aug(track):

    bDesired = True

    if (track.fVelocity < 34.7 or track.fVelocity > 97.7):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Sep
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of September
def ComputeTrackQuality_Velocity_Sep(track):

    bDesired = True

    if (track.fVelocity < 77 or track.fVelocity > 100):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Oct
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of October
def ComputeTrackQuality_Velocity_Oct(track):

    bDesired = True

    if (track.fVelocity < 59 or track.fVelocity > 90):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Nov
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of November
def ComputeTrackQuality_Velocity_Nov(track):

    bDesired = True

    if (track.fVelocity < 43 or track.fVelocity > 75):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity_Dec
# Input:
#   track: A given track to be evaluated
# Output:
#   Whether or not the given track has a statistically likely velocity for the month of December
def ComputeTrackQuality_Velocity_Dec(track):

    bDesired = True

    if (track.fVelocity < 44 or track.fVelocity > 83):
        bDesired = False

    return bDesired

# ComputeTrackQuality_Velocity
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.bFlaggedVelocity
def ComputeTrackQuality_Velocity(track):

    # By default
    bHasDesiredVelocity = False

    idx = track.nMonthIndex

    if (0 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Jan(track)
    elif (1 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Feb(track)
    elif (2 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Mar(track)
    elif (3 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Apr(track)
    elif (4 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_May(track)
    elif (5 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Jun(track)
    elif (6 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Jul(track)
    elif (7 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Aug(track)
    elif (8 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Sep(track)
    elif (9 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Oct(track)
    elif (10 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Nov(track)
    elif (11 == idx):
        bHasDesiredVelocity = ComputeTrackQuality_Velocity_Dec(track)

    if (bHasDesiredVelocity):
        return

    track.bFlaggedVelocity = True

# ComputeTrackQuality_GridDirection
# Input:
#   track: A given track to be evaluated
# Modifies:
#   track.bFlaggedGridDirection
def ComputeTrackQuality_GridDirection(track):

    idx = track.nGridSection
    bHasDesiredGridDirection = False

    slen = len(m_listGridDirections[idx])

    for t in range(slen):

        dir1 = m_listGridDirections[idx][t][0]
        dir2 = m_listGridDirections[idx][t][1]

        if (track.fDirection >= dir1 and track.fDirection <= dir2):
            bHasDesiredGridDirection = True
            break

    if (bHasDesiredGridDirection):
        return

    track.bFlaggedGridDirection = True
