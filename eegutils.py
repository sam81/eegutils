# -*- coding: utf-8 -*-
#   Copyright (C) 2012-2013 Samuele Carcagno <sam.carcagno@gmail.com>
#   This file is part of eegutils

#    eegutils is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    eegutils is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with eegutils.  If not, see <http://www.gnu.org/licenses/>.

"""
This module contains functions to extract and process event related potentials (ERPs) from
electroencephalographic (EEG) recordings.
"""

from __future__ import division
import copy, numpy
from numpy import abs, append, arange, array, array_equal, convolve, ceil, floor, diff, mean, repeat, where, zeros
from numpy.fft import fft
from scipy import signal
from scipy.signal import firwin2, blackman, hamming, hanning, bartlett, fftconvolve
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame
try:
    import biosig
except ImportError:
    pass
import ctypes

__version__ = "0.0.3"

def averageAverages(aveList, nSegments):
    """
    Perform a weighted average of a list of averages. The weight of
    each average in the list is determined by the number of segments
    from which it was obtained. 
    
    Parameters
    ----------
    aveList : list of dicts of 2D numpy arrays
        The list of averages for each experimental condition.
    nSegments : list of dicts of ints
        The number of epochs on which each average is based.

    Returns
    ----------
    weightedAve : dict of 2D numpy arrays
        The weighted averages for each condition.
    nSegsSum : dict of ints
        The number of epochs on which each weighted average is based.

    Examples
    ----------
    >>> #simulate averages
    >>> import numpy as np
    >>> ave1 = {'cnd1': np.random.rand(4, 2048), 'cnd2': np.random.rand(4, 2048)}
    >>> ave2 = {'cnd1': np.random.rand(4, 2048), 'cnd2': np.random.rand(4, 2048)}
    >>> nSegs1 = {'cnd1': 196, 'cnd2': 200}
    >>> nSegs2 = {'cnd1': 198, 'cnd2': 189}
    >>> aveList = [ave1, ave2]; nSegments = [nSegs1, nSegs2]
    >>> weightedAve, nSegsSum = averageAverages(aveList=aveList, nSegments=nSegments)
    """
    eventList = list(aveList[0].keys())
    nSegsSum = {}
    weightedAve = {}
    for event in eventList:
        nSegsSum[event] = 0
        for i in range(len(aveList)):
            nSegsSum[event] = nSegsSum[event] + nSegments[i][event]

    for event in eventList:
        weightedAve[event] = numpy.zeros(aveList[0][event].shape, dtype=aveList[0][eventList[0]].dtype)
        for i in range(len(aveList)):
           weightedAve[event] = weightedAve[event] + aveList[i][event] * (nSegments[i][event]/nSegsSum[event])
    
    return weightedAve, nSegsSum

def averageEpochs(rec):
    """
    Average the epochs of a segmented recording.

    Parameters
    ----------
    rec : dict of 3D numpy arrays with dimensions (n_channels x n_samples x n_epochs)
        The segmented recording

    Returns
    ----------
    ave : dict of 2D numpy arrays with dimensions (n_channels x n_samples)
        The average epochs for each condition.
    nSegs : dict of ints
        The number of epochs averaged for each condition.
        
    Examples
    ----------
    >>> ave, nSegs = averageEpochs(rec=rec)
    """
    
    eventList = list(rec.keys())
    ave = {}
    nSegs = {}
    for code in eventList:
        nSegs[code] = rec[code].shape[2]
        ave[code] = numpy.mean(rec[code], axis=2)

    return ave, nSegs

def baselineCorrect(rec, baselineStart, preDur, sampRate):
    """
    Perform baseline correction by subtracting the average pre-event
    voltage from each channel of a segmented recording.

    Parameters
    ----------
    rec: dict of 3D arrays
        The segmented recording.
    baselineStart: float
        Start time of the baseline window relative to the event onset, in seconds.
        The absolute value of baselineStart cannot be greater than preDur.
        In practice baselineStart allows you to define a baseline window shorter
        than the time window before the experimental event (preDur).
    preDur: float
        Duration of recording before the experimental event, in seconds.
    sampRate: int
        The samplig rate of the EEG recording.
    
    Examples
    ----------
    #baseline window has the same duration of preDur
    >>> baseline_correct(rec=rec, baselineStart=-0.2, preDur=0.2, sampRate=512)
    #now with a baseline shorter than preDur
    >>> baseline_correct(rec=rec, baselineStart=-0.15, preDur=0.2, sampRate=512)
    """
    eventList = list(rec.keys())
    epochStartSample = int(round(preDur*sampRate))
    baselineStartSample = int(epochStartSample - abs(round(baselineStart*sampRate)))

   
    for i in range(len(eventList)): #for each event
        for j in range(rec[str(eventList[i])].shape[2]): #for each epoch
            for k in range(rec[str(eventList[i])].shape[0]): #for each electrode
                thisBaseline = numpy.mean(rec[str(eventList[i])][k,baselineStartSample:epochStartSample,j])
                rec[str(eventList[i])][k,:,j] = rec[str(eventList[i])][k,:,j] - thisBaseline
    return 
    
def chainSegments(rec, nChunks, sampRate, start=None, end=None, baselineDur=0):
    """
    Take a dictionary containing in each key a list of segments, and chain these segments
    into chunks of length nChunks
    baselineDur is for determining what is the zero point
    start and end are given with reference to the zero point
    """
    baseline_pnts = round(baselineDur * sampRate)
    startPnt = int(round(start*sampRate) + baseline_pnts) 
    endPnt = int(round(end*sampRate) + baseline_pnts)
    chunk_size = endPnt - startPnt
    sweep_size = chunk_size * nChunks
    nReps = {}
    eventList = list(rec.keys())
    eegChained = {}
    fromeegChainedAve = {}
    for i in range(len(eventList)):
        currCode = eventList[i]
        eegChained[currCode] = zeros((rec[currCode].shape[0], sweep_size), dtype=rec[currCode].dtype)  #two-dimensional array of zeros
        fromeegChainedAve[currCode] = zeros((rec[currCode].shape[0], chunk_size), dtype=rec[currCode].dtype)
        nReps[currCode] = zeros((nChunks))
        p = 0
        k = 0
        while k < rec[currCode].shape[2]:
            if p > (nChunks-1):
                p = 0
            
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] + rec[currCode][:,startPnt:endPnt, k]
            nReps[currCode][p] = nReps[currCode][p] + 1
            fromeegChainedAve[currCode] = fromeegChainedAve[currCode] + rec[currCode][:,startPnt:endPnt, k]
            p = p+1 #p is the chunk counter
            k = k+1 #k is the epoch counter

    for i in range(len(eventList)):
        currCode = eventList[i]
        for p in range(nChunks):
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] / nReps[currCode][p]
        fromeegChainedAve[currCode] = fromeegChainedAve[currCode] / sum(nReps[currCode])
    return eegChained



def detrend(rec):
    """
    
    """
    nChannels = rec.shape[0]
    for i in range(nChannels):
        rec[i,:] = rec[i,:] - numpy.mean(rec[i,:])
    return rec

def detrendSegmented(rec):
    """
    
    """
    eventList = list(rec.keys())
    for ev in eventList:
        for i in range(len(rec[ev])):
            for j in range(rec[ev][0].shape[0]):
                rec[ev][i][j,:] = rec[ev][i][j,:] - numpy.mean(rec[ev][i][j,:])
    return(rec)


def extractEventTable(trigChan, sampRate):
    trigst = copy.copy(trigChan)
    trigst[where(diff(trigst) == 0)[0]+1] = 0
    startPoints = where(trigst != 0)[0]
    
    trige = diff(trigChan)
    stopPoints = where(trige != 0)[0]
    stopPoints = append(stopPoints, len(trigChan)-1)
    trigDurs = (stopPoints - startPoints)/sampRate

    evt = trigst[where(trigst != 0)]

    eventTable = {}
    eventTable['code'] = evt
    eventTable['idx'] = startPoints
    #eventTable['stop_idx'] = stopPoints
    eventTable['dur'] = trigDurs

    return eventTable
    

def filterSegmented(rec, channels, sampRate, filterType, nTaps, cutoffs, transitionWidth):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    
    eventList = list(rec.keys())

    nChannels = rec[eventList[0]][0].shape[0]
    if channels == None or len(channels) == 0:
        channels = list(range(nChannels))
   
    if filterType == "lowpass":
        f3 = cutoffs[0]
        f4 = cutoffs[0] * (1+transitionWidth)
        f3 = (f3*2) / sampRate
        f4 = (f4*2) / sampRate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filterType == "highpass":
        f1 = cutoffs[0] * (1-transitionWidth)
        f2 = cutoffs[0]
        f1 = (f1*2) / sampRate
        f2 = (f2*2) / sampRate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filterType == "bandpass":
        f1 = cutoffs[0] * (1-transitionWidth)
        f2 = cutoffs[0]
        f3 = cutoffs[1]
        f4 = cutoffs[1] * (1+transitionWidth)
        f1 = (f1*2) / sampRate
        f2 = (f2*2) / sampRate
        f3 = (f3*2) / sampRate
        f4 = (f4*2) / sampRate
        f = [0, f1, f2, ((f2+f3)/2), f3, f4, 1]
        m = [0, 0.00003, 1, 1, 1, 0.00003, 0]
    b = firwin2 (nTaps,f,m);
    ## w,h = signal.freqz(b,1)
    ## h_dB = 20 * log10 (abs(h))
    ## plt.plot((w/max(w))*(sampRate/2),h_dB)
    ## plt.show()

    
    for ev in eventList:
        for i in range(rec[ev].shape[2]): #for each epoch
            for j in range(rec[ev].shape[0]): #for each channel
                if j in channels:
                    rec[ev][j,:,i] = fftconvolve(rec[ev][j,:,i], b, 'same')
                    rec[ev][j,:,i] = fftconvolve(rec[ev][j,:,i][::-1], b, 'same')[::-1]
    return 
        
def filterContinuous(rec, channels, sampRate, filterType, nTaps, cutoffs, transitionWidth):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
       
    if filterType == "lowpass":
        f1 = cutoffs[0] * (1-transitionWidth)
        f2 = cutoffs[0]
        f1 = (f1*2) / sampRate
        f2 = (f2*2) / sampRate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filterType == "highpass":
        f1 = cutoffs[0] * (1-transitionWidth)
        f2 = cutoffs[0]
        f1 = (f1*2) / sampRate
        f2 = (f2*2) / sampRate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filterType == "bandpass":
        f1 = cutoffs[0] * (1-transitionWidth)
        f2 = cutoffs[0]
        f3 = cutoffs[1]
        f4 = cutoffs[1] * (1+transitionWidth)
        f1 = (f1*2) / sampRate
        f2 = (f2*2) / sampRate
        f3 = (f3*2) / sampRate
        f4 = (f4*2) / sampRate
        f = [0, f1, f2, ((f2+f3)/2), f3, f4, 1]
        m = [0, 0.00003, 1, 1, 1, 0.00003, 0]
    b = firwin2 (nTaps,f,m);
    b = b.astype(rec.dtype)
    #print(b[0:3])
    nChannels = rec.shape[0]
    if channels == None:
        channels = list(range(nChannels))

    for i in range(nChannels):
        if i in channels:
            rec[i,:] = fftconvolve(rec[i,:], b, "same")
            rec[i,:] = fftconvolve(rec[i,:][::-1], b, "same")[::-1]
    return(rec)

def findArtefactThresh(rec, thresh=[100], channels=[0]):
    """
    Find epochs with voltage values exceeding a given threshold.
    
    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented recording
    thresh :
        The threshold value.
    channels = array or list of ints
        The indexes of the channels on which to find artefacts.
        
    Returns
    ----------

    Examples
    ----------
    """
    if len(channels) != len(thresh):
        print("The number of thresholds must be equal to the number of channels")
        return
    eventList = list(rec.keys())
    segsToReject = {}
    for i in range(len(eventList)):
        segsToReject[str(eventList[i])] = [] #list to keep the indices of the epochs to delete
        for j in range(rec[str(eventList[i])].shape[2]): #for each epoch
            for k in range(rec[str(eventList[i])].shape[0]): #for each channel
                if k in channels:
                    if (max(rec[str(eventList[i])][k,:,j]) > thresh[channels.index(k)] or min(rec[str(eventList[i])][k,:,j]) < -thresh[channels.index(k)]) == True:
                        segsToReject[str(eventList[i])].append(j)
                

    for i in range(len(eventList)): #segment may be flagged by detection in more than one channel
        segsToReject[str(eventList[i])] = numpy.unique(segsToReject[str(eventList[i])])

    return segsToReject


def getFRatios(ffts, compIdx, nSideComp, nExcludedComp, otherExclude):
    """
    Add excluded components
    """
    cnds = ffts.keys()
    fftVals = {}
    fRatio = {}
    dfNum = 2
    dfDenom = 2*(nSideComp*2) -1
    for cnd in cnds:
        fRatio[cnd] = {}
        fftVals[cnd] = {}
        fRatio[cnd]['F'] = []
        fRatio[cnd]['pval'] = []
        fftVals[cnd]['sigPow'] = []
        fftVals[cnd]['noisePow'] = []
        for c in range(len(compIdx)):
            sideBands = getNoiseSidebands(compIdx, nSideComp, nExcludedComp, ffts[cnd]['mag'], otherExclude)
            noisePow = numpy.mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][compIdx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio

def getNoiseSidebands(components, nCompSide, nExcludeSide, FFTArray, otherExclude=None):
    """
    the 2 has the possibility to exclude extra components, useful for distortion products
    """
    #components: a list containing the indexes of the target components
    #nCompSide: number of components used for each side band
    #nExcludeSide: number of components adjacent to to the target components to exclude
    #FFTArray: array containing the fft values
    idxProtect = []; idxProtect.extend(components);
    if otherExclude != None:
        idxProtect.extend(otherExclude)
    for i in range(nExcludeSide):
        idxProtect.extend(numpy.array(components) + (i+1))
        idxProtect.extend(numpy.array(components) - (i+1))
    #idxProtect = sorted(idxProtect)
    #print(idxProtect)

    noiseBands = []
    for i in range(len(components)):
        loSide = []
        hiSide = []
        counter = 1
        while len(hiSide) < nCompSide:
            currIdx = components[i] + nExcludeSide + counter
            if currIdx not in idxProtect:
                hiSide.append(FFTArray[currIdx])
            counter = counter + 1
        counter = 1
        while len(loSide) < nCompSide:
            currIdx = components[i] - nExcludeSide - counter
            if currIdx not in idxProtect:
                loSide.append(FFTArray[currIdx])
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        
    return noiseBands


def mergeTriggersCnt(trigArray, trigList, newTrig):
    """
    Take one or more triggers in trig_list, and substitute them with new_trig

    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    
    trigArray[numpy.in1d(trigArray, trigList)] = newTrig

    return 

def mergeTriggersEventTable(eventTable, trigList, newTrig):
    """
    Substitute the event table triggers listed in trigList
    with newTrig

    Parameters
    ----------
    eventTable : dict of int arrays
        The event table
    trigList : array of ints
        The list of triggers to substitute
    newTrig : int
        The new trigger used to substitute the triggers
        in trigList

    Returns
    ----------

    Examples
    ----------
    """
    
    eventTable['code'][numpy.in1d(eventTable['code'], trigList)] = newTrig
   
    return 

def read_biosig(fileName):
    """
    Wrapper of biosig4python functions for reading Biosemi BDF files.
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    HDR = biosig.constructHDR(0,0)
    HDR = biosig.sopen(fileName, 'r', HDR)
    data = biosig.sread(0, HDR.NRec, HDR)

    if HDR.EVENT.TYP:
        TYP = ctypes.cast( HDR.EVENT.TYP.__long__(), ctypes.POINTER( ctypes.c_uint16 ) )
    if HDR.EVENT.CHN:
        CHN = ctypes.cast( HDR.EVENT.CHN.__long__(), ctypes.POINTER( ctypes.c_uint16 ) )
    if HDR.EVENT.POS:
        POS = ctypes.cast( HDR.EVENT.POS.__long__(), ctypes.POINTER( ctypes.c_uint32 ) )
    if HDR.EVENT.DUR:
        DUR = ctypes.cast( HDR.EVENT.DUR.__long__(), ctypes.POINTER( ctypes.c_uint32 ) )

    codes = []
    pos = []
    dur = []
    for k in range(HDR.EVENT.N):
        codes.append(TYP[k] & (256-1))
        pos.append(int(POS[k]))
        dur.append(int(DUR[k]))
        

 
    # close file
    biosig.sclose(HDR)
    #
    # release allocated memory
    biosig.destructHDR(HDR)
    bdfRec = {}
    event_table = {}
    event_table['trigs'] = array(codes)
    event_table['start_idx'] = array(pos)
    #event_table['stop_idx'] = stopPoints
    event_table['trigs_dur'] = array(dur)
    return data, event_table

def removeEpochs(rec, toRemove):
    """
    Remove epochs from a segmented recording.
    
    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented recording
    to_remove : dict of 1D arrays
        List of epochs to remove for each condition

    Returns
    ----------

    Examples
    ----------
    """
    eventList = list(rec.keys())
    for code in eventList:
        rec[code] = numpy.delete(rec[code], toRemove[code], axis=2)
    
    return 

## def removeSpuriousTriggers(eventTable, sentTrigs, minInt, sampRate):
##     """
##     Remove spurious trigger codes.

##     Parameters
##     ----------
##     eventTable :  dict with the following keys
##        - code: array of ints
##            The list of triggers in the EEG recording.
##        - idx : array of ints
##            The indexes of trigs in the EEG recording.
##     sent_triggers : array of floats
##         Array containing the list of triggers that were sent to the EEG recording equipment.
##     minInt : float
##         The minimum possible time interval between consecutive triggers in seconds
##     sampRate : int
##         The sampling rate of the EEG recording


##     Returns
##     -------
##     eventTable :  dict with the following keys
##        - trigs: array of ints
##           List of valid triggers.
##        - trigs_pos : array of ints
##           The indexes of trigs in the EEG recording

##     res_info: dict with the following keys:
##        - len_matching: int
##           Number of matching elements in the event table and sentTrigs
##        - len_sent: int
##           Length of sentTrigs
##        - match : boolean
##           True if a sequence matching the sentTrigs sequence is found in the eventTable
    
##     Examples
##     --------
##     >>> 
##     ... 
##     >>> 
##     ... 
##     >>> 
##     """
##     rec_trigs = eventTable['code']
##     rec_trigs_idx = eventTable['idx']

##     allowed_trigs = numpy.unique(sentTrigs)
##     rec_trigs_idx = rec_trigs_idx[numpy.in1d(rec_trigs, allowed_trigs)]
##     rec_trigs = rec_trigs[numpy.in1d(rec_trigs, allowed_trigs)]

##     intervals_ok = False
##     while intervals_ok == False:
##         intervals = numpy.diff(rec_trigs_idx) / sampRate
##         intervals = numpy.insert(intervals, 0, minInt+1)
##         if intervals[intervals < minInt].shape[0] == 0:
##             intervals_ok = True
##         else:
##             idx_to_del = (numpy.where(intervals<minInt)[0][0])
##             #print(rec_trigs_idx)
##             rec_trigs = numpy.delete(rec_trigs, idx_to_del)
##             rec_trigs_idx = numpy.delete(rec_trigs_idx, idx_to_del)
  
   
   
##     if numpy.array_equal(rec_trigs, sentTrigs) == True:
##         match_found = True
##     else:
##         match_found = False

##     eventTable['code'] = rec_trigs
##     eventTable['idx'] = rec_trigs_idx

##     res_info = {}
##     res_info['match'] = match_found
##     res_info['len_sent'] = len(sentTrigs)
##     res_info['len_selected'] = len(rec_trigs)

##     return res_info

def removeSpuriousTriggers(eventTable, sentTrigs, minTrigDur):
    rec_trigs = eventTable['code']
    rec_trigs_dur = eventTable['dur']
    rec_trigs_start = eventTable['idx']
    #rec_trigs_stop = eventTable['stop_idx']
    
    allowed_trigs = numpy.unique(sentTrigs)
    rec_trigs_dur = rec_trigs_dur[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs_start = rec_trigs_start[numpy.in1d(rec_trigs, allowed_trigs)]
    #rec_trigs_stop = rec_trigs_stop[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs = rec_trigs[numpy.in1d(rec_trigs, allowed_trigs)]

    rec_trigs = rec_trigs[rec_trigs_dur >= minTrigDur]
    rec_trigs_start = rec_trigs_start[rec_trigs_dur >= minTrigDur]
    #rec_trigs_stop = rec_trigs_stop[rec_trigs_dur >= minTrigDur]
    rec_trigs_dur = rec_trigs_dur[rec_trigs_dur >= minTrigDur]

    if numpy.array_equal(rec_trigs, sentTrigs) == True:
        match_found = True
    else:
        match_found = False

    #x = diff(rec_trigs_start)/2048
    #print(x[x<1.375])
    #print(min(x), max(x), mean(x))
    eventTable['code'] = rec_trigs
    eventTable['dur'] = rec_trigs_dur
    eventTable['idx'] = rec_trigs_start
    #eventTable['stop_idx'] = rec_trigs_stop

    res_info = {}
    res_info['match'] = match_found
    res_info['len_sent'] = len(sentTrigs)
    res_info['len_found'] = len(rec_trigs)

    return res_info

def rerefCnt(rec, refChannel, channels=None):
    """
    Rereference channels in a continuous recording.

    Parameters
    ----------
    rec : 
        Recording
    refChannel: int
        The reference channel (indexing starts from zero).
    channels : list of ints
        List of channels to be rereferenced (indexing starts from zero).
  
    Returns
    -------
    rec : an array of floats with dimenions nChannels X nDataPoints
        
    Examples
    --------
    >>> rerefCnt(rec=dats, refChannel=4, channels=[1, 2, 3])
    """

    if channels == None:
        nChannels = rec.shape[0]
        channels = list(range(nChannels))

    rec[channels,:] = rec[channels,:] - rec[refChannel,:]

    return 



def segmentCnt(rec, eventTable, epochStart, epochEnd, sampRate, eventList=None):
    """
    Segment a continuous EEG recording into discrete event-related epochs.
    
    Parameters
    ----------
    rec: array of floats
        The EEG data.
    eventTable : dict with the following keys
       - trigs : array of ints
           The list of triggers in the EEG recording.
       - trigs_pos : array of ints
           The indexes of trigs in the EEG recording.
    epochStart : float
        The time at which the epoch starts relative to the trigger code, in seconds.
    epochEnd : float
        The time at which the epoch ends relative to the trigger code, in seconds.
    sampRate : int
        The sampling rate of the EEG recording.
    eventList : list of ints
        The list of events for which epochs should be extracted.
        If no list is given epochs will be extracted for all the trigger
        codes present in the event table.
    
    Returns
    ----------
    segs : dict of 3D arrays
        The segmented recording. The dictionary has a key for each condition.
        The corresponding key value is a 3D array with dimensions
        nChannels x nSamples x nSegments
    n_segs : dict of ints
        The number of segments for each condition.
        
    Examples
    ----------
    >>>  segs, n_segs = eeg.segment_cnt(rec=dats, eventTable=evt_tab, epochStart=-0.2, epochEnd=0.8, sampRate=512, eventList=['200', '201'])
    """
    trigs = eventTable['code']
    if eventList == None:
        eventList = numpy.unique(trigs)

    trigs = eventTable['code']
    trigs_pos = eventTable['idx']
    epochStartSample = int(round(epochStart*sampRate))
    epochEndSample = int(round(epochEnd*sampRate))

    nSamples = epochEndSample - epochStartSample
    segs = {}
    for i in range(len(eventList)):
        idx = trigs_pos[numpy.where(trigs == eventList[i])[0]]
        segs[str(eventList[i])] = numpy.zeros((rec.shape[0], nSamples, len(trigs[trigs==eventList[i]])), dtype=rec.dtype)
        for j in range(len(idx)):
            thisStartPnt = (idx[j]+epochStartSample)
            #print(thisStartPnt)
            thisStopPnt = (idx[j]+epochEndSample)
            if thisStartPnt < 0 or thisStopPnt > rec.shape[1]:
                if thisStartPnt < 0:
                    print(idx[j], "Epoch starts before start of recording. Skipping")
                if thisStopPnt > rec.shape[1]:
                    print(idx[j], "Epoch ends after end of recording. Skipping")
            else:
                segs[str(eventList[i])][:,:,j] = rec[:, thisStartPnt:thisStopPnt]
    n_segs = {}
    for i in range(len(eventList)): #count
            n_segs[str(eventList[i])] = segs[str(eventList[i])].shape[2]

    return segs, n_segs



        


#Utility functions
#############
def nextPowTwo(x):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    out = int(ceil(log2(x)))
    return out

def getFFT(sig, sampRate, window, powerOfTwo):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if powerOfTwo == True:
        nfft = 2**nextPowTwo(n)
    else:
        nfft = n
    if window != 'none':
        if window == 'hamming':
             w = hamming(n)
        elif window == 'hanning':
             w = hanning(n)
        elif window == 'blackman':
             w = blackman(n)
        elif window == 'bartlett':
             w = bartlett(n)
        sig = sig*w
        
    p = fft(sig, nfft) # take the fourier transform 
    nUniquePts = ceil((nfft+1)/2.0)
    p = p[0:nUniquePts]
    p = abs(p)
    p = p / sampRate  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    #p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampRate / nfft);
    x = {'freqArray': freqArray, 'mag':p}
    return x

def getSpectrogram(sig, sampRate, winLength, overlap, winType, powerOfTwo):
    """
    winLength in seconds
    overlap in percent
    if the signal length is not a multiple of the window length it is trucated
    """
    winLengthPnt = floor(winLength * sampRate)
    step = winLengthPnt - round(winLengthPnt * overlap / 100.)
    ind = arange(0, len(sig) - winLengthPnt, step)
    n = len(ind)

    x = getSpectrum(sig[ind[0]:ind[0]+winLengthPnt], sampRate, winType, powerOfTwo)
    freq_array = x['freq']; p = x['mag']

    power_matrix = zeros((len(freq_array), n))
    power_matrix[:,0] = p
    for i in range(1, n):
        x = get_spectrum(sig[ind[i]:ind[i]+winLengthPnt], sampRate, winType, powerOfTwo)
        freq_array = x['freq']; p = x['mag']
        power_matrix[:,i] = p

    timeInd = arange(0, len(sig), step)
    time_array = 1./sampRate * (timeInd)
    x = {'freq': freq_array, 'time': time_array, 'mag': power_matrix}
    return x

def getSpectrum(sig, sampRate, window, powerOfTwo):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if powerOfTwo == True:
        nfft = 2**nextPowTwo(n)
    else:
        nfft = n
    if window != 'none':
        if window == 'hamming':
             w = hamming(n)
        elif window == 'hanning':
             w = hanning(n)
        elif window == 'blackman':
             w = blackman(n)
        elif window == 'bartlett':
             w = bartlett(n)
        sig = sig*w
    #print(sig[0:2])
    p = fft(sig, nfft) # take the fourier transform 
    nUniquePts = ceil((nfft+1)/2)
    p = p[0:nUniquePts]
    p = abs(p)
    p = p / sampRate  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freq_array = arange(0, nUniquePts, 1.0) * (sampRate / nfft);
    x = {'freq': freq_array, 'mag':p}
    return x


