#!/usr/bin/env python
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
from scipy.signal import firwin2, blackman, hamming, hanning, bartlett
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame
try:
    import biosig
except ImportError:
    pass
import ctypes

__version__ = "0.1.2"

def average_averages(ave_list, n_segments):
    """
    Perform a weighted average of a list of averages. The weight of
    each average in the list is determined by the number of segments
    from which it was obtained.
    
    Parameters
    ----------
    ave_list : dict of list of 2D numpy arrays
        The list of averages for each experimental condition.
    n_segments : dict of ints
        The number of epochs on which each average is based 

    Returns
    ----------

    Examples
    ----------
    """
    eventList = list(ave_list[0].keys())
    nSegsSum = {}
    weightedAve = {}
    for event in eventList:
        nSegsSum[event] = 0
        for i in range(len(ave_list)):
            nSegsSum[event] = nSegsSum[event] + n_segments[i][event]

    for event in eventList:
        weightedAve[event] = numpy.zeros(ave_list[0][event].shape)
        for i in range(len(ave_list)):
           weightedAve[event] = weightedAve[event] + ave_list[i][event] * (n_segments[i][event]/nSegsSum[event])
    
    return weightedAve, nSegsSum

def average_epochs(rec):
    """
    Average the epochs of a segmented recording.

    Parameters
    ----------
    rec : dict of 3D numpy arrays with dimensions (n_channels x n_samples x n_epochs)
        Recording

    Returns
    ----------
    ave : dict of 2D numpy arrays with dimensions (n_channels x n_samples)
        The average epochs for each condition.
    n_segs : dict of ints
        The number of epochs averaged for each condition.
        
    Examples
    ----------
    >>> ave, n_segs = average_epochs(rec=rec)
    """
    
    eventList = list(rec.keys())
    ave = {}
    n_segs = {}
    for code in eventList:
        n_segs[code] = rec[code].shape[2]
        ave[code] = numpy.mean(rec[code], axis=2)

    return ave, n_segs

def baseline_correct(rec, baseline_start, pre_dur, samp_rate):
    """
    Perform baseline correction by subtracting the average pre-event
    voltage from each channel of a segmented recording.

    Parameters
    ----------
    rec: dict of 3D arrays
        The segmented recording.
    baseline_start: float
        Start time of the baseline window relative to the event onset, in seconds.
        The absolute value of baseline_start cannot be greater than pre_dur.
        In practice baseline_start allows you to define a baseline window shorter
        than the time window before the experimental event (pre_dur).
    pre_dur: float
        Duration of recording before the experimental event, in seconds.
    samp_rate: int
        The samplig rate of the EEG recording.
    
    Examples
    ----------
    #baseline window has the same duration of pre_dur
    >>> baseline_correct(rec=rec, baseline_start=-0.2, pre_dur=0.2, samp_rate=512)
    #now with a baseline shorter than pre_dur
    >>> baseline_correct(rec=rec, baseline_start=-0.15, pre_dur=0.2, samp_rate=512)
    """
    eventList = list(rec.keys())
    epochStartSample = int(round(pre_dur*samp_rate))
    baseline_startSample = int(epochStartSample - abs(round(baseline_start*samp_rate)))
   
    for i in range(len(eventList)): #for each event
        for j in range(rec[str(eventList[i])].shape[2]): #for each epoch
            for k in range(rec[str(eventList[i])].shape[0]): #for each electrode
                thisBaseline = numpy.mean(rec[str(eventList[i])][k,baseline_startSample:epochStartSample,j])
                rec[str(eventList[i])][k,:,j] = rec[str(eventList[i])][k,:,j] - thisBaseline
    return 
    
def chain_segments(rec, n_chunks, samp_rate, start=None, end=None, baseline_dur=0):
    """
    Take a dictionary containing in each key a list of segments, and chain these segments
    into chunks of length n_chunks
    baseline_dur is for determining what is the zero point
    start and end are given with reference to the zero point
    """
    baseline_pnts = round(baseline_dur * samp_rate)
    startPnt = int(round(start*samp_rate) + baseline_pnts) 
    endPnt = int(round(end*samp_rate) + baseline_pnts) 
    chunk_size = endPnt - startPnt
    sweep_size = chunk_size * n_chunks
    nReps = {}
    eventList = list(rec.keys())
    eegChained = {}
    fromeegChainedAve = {}
    for i in range(len(eventList)):
        currCode = eventList[i]
        eegChained[currCode] = zeros((rec[currCode].shape[0], sweep_size))  #two-dimensional array of zeros
        fromeegChainedAve[currCode] = zeros((rec[currCode].shape[0], chunk_size))
        nReps[currCode] = zeros((n_chunks))
        p = 0
        k = 0
        while k < rec[currCode].shape[2]:
            if p > (n_chunks-1):
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
        for p in range(n_chunks):
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] / nReps[currCode][p]
        fromeegChainedAve[currCode] = fromeegChainedAve[currCode] / sum(nReps[currCode])
    return eegChained

def combine_chained(d_list):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    cnds = d_list[0].keys()
    cmb = {}
    for cnd in cnds:
        for i in range(len(d_list)):
            if i == 0:
                cmb[cnd] = d_list[0][cnd]
            else:
                cmb[cnd] = cmb[cnd] + d_list[i][cnd]
        cmb[cnd] = cmb[cnd] / len(d_list)
            
    return cmb

def detrend(rec):
    """
    
    """
    nChannels = rec.shape[0]
    for i in range(nChannels):
        rec[i,:] = rec[i,:] - numpy.mean(rec[i,:])
    return rec

def detrend_segmentsed(rec):
    """
    
    """
    eventList = list(rec.keys())
    for ev in eventList:
        for i in range(len(rec[ev])):
            for j in range(rec[ev][0].shape[0]):
                rec[ev][i][j,:] = rec[ev][i][j,:] - numpy.mean(rec[ev][i][j,:])
    return(rec)

def extract_event_table(trig_chan, null_trig=0):
    """
    Extract the event table from the channel containing the trigger codes.

    Parameters
    ----------
    trig_chan : array of ints
        The trigger channel.

    Returns
    -------
    event_table :  dict with the following keys
       - trigs: array of ints
          The list of triggers in the EEG recording.
       - trigs_pos : array of ints
          The indexes of the triggers in the EEG recording.
    
    Examples
    --------
    >>> 
    ... 
    >>> 
    ... 
    >>> 
    """
    trigs = trig_chan[trig_chan!=null_trig]
    trigs_idx = numpy.where((trig_chan!=null_trig))[0]

    evtTable = {}
    evtTable['trigs'] = trigs
    evtTable['trigs_idx'] = trigs_idx
    
    return evtTable

def extract_event_table_nonorm(trig_chan, samp_rate):
    trigst = copy.copy(trig_chan)
    trigst[where(diff(trigst) == 0)[0]+1] = 0
    startPoints = where(trigst != 0)[0]
    
    trige = diff(trig_chan)
    stopPoints = where(trige != 0)[0]
    stopPoints = append(stopPoints, len(trig_chan)-1)
    trigDurs = (stopPoints - startPoints)/samp_rate

    evt = trigst[where(trigst != 0)]

    event_table = {}
    event_table['trigs'] = evt
    event_table['start_idx'] = startPoints
    event_table['stop_idx'] = stopPoints
    event_table['trigs_dur'] = trigDurs

    return event_table
    

def filter_segmented(rec, channels, samp_rate, filter_type, n_taps, cutoffs, transition_width):
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
   
    if filter_type == "lowpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filter_type == "highpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filter_type == "bandpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f3 = cutoffs[1]
        f4 = cutoffs[1] * (1+transition_width)
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f3 = (f3*2) / samp_rate
        f4 = (f4*2) / samp_rate
        f = [0, f1, f2, ((f2+f3)/2), f3, f4, 1]
        m = [0, 0.00003, 1, 1, 1, 0.00003, 0]
    b = firwin2 (n_taps,f,m);
    ## w,h = signal.freqz(b,1)
    ## h_dB = 20 * log10 (abs(h))
    ## plt.plot((w/max(w))*(samp_rate/2),h_dB)
    ## plt.show()

    
    for ev in eventList:
        for i in range(rec[ev].shape[2]): #for each epoch
            for j in range(rec[ev].shape[0]): #for each channel
                if j in channels:
                    rec[ev][j,:,i] = convolve(rec[ev][j,:,i], b, 'same')
                    rec[ev][j,:,i] = convolve(rec[ev][j,:,i][::-1], b, 'same')[::-1]
    return 
        
def filter_continuous(rec, channels, samp_rate, filter_type, n_taps, cutoffs, transition_width):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
       
    if filter_type == "lowpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filter_type == "highpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filter_type == "bandpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f3 = cutoffs[1]
        f4 = cutoffs[1] * (1+transition_width)
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f3 = (f3*2) / samp_rate
        f4 = (f4*2) / samp_rate
        f = [0, f1, f2, ((f2+f3)/2), f3, f4, 1]
        m = [0, 0.00003, 1, 1, 1, 0.00003, 0]
    b = firwin2 (n_taps,f,m);

    nChannels = rec.shape[0]
    if channels == None:
        channels = list(range(nChannels))
   
    for i in range(nChannels):
        if i in channels:
            rec[i,:] = convolve(rec[i,:], b, "same")
            rec[i,:] = convolve(rec[i,:][::-1], b,1)[::-"same"]
    return(rec)

def find_artefact_thresh(rec, thresh_lower=[-100], thresh_higher=[100], channels=None):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    eventList = list(rec.keys())
    segs_to_reject = {}
    for i in range(len(eventList)):
        segs_to_reject[str(eventList[i])] = [] #list to keep the indices of the epochs to delete
        for j in range(rec[str(eventList[i])].shape[2]): #for each epoch
            for k in range(rec[str(eventList[i])].shape[0]): #for each channel
                if k in channels:
                    if (max(rec[str(eventList[i])][k,:,j]) > thresh_higher[channels.index(k)] or min(rec[str(eventList[i])][k,:,j]) < thresh_lower[channels.index(k)]) == True:
                        segs_to_reject[str(eventList[i])].append(j)
                
            
    for i in range(len(eventList)): #segment may be flagged by detection in more than one channel
        segs_to_reject[str(eventList[i])] = numpy.unique(segs_to_reject[str(eventList[i])])

    return segs_to_reject

def get_F_ratios(ffts, comp_idx, n_side_comp, n_excluded_comp):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    cnds = ffts.keys()
    fftVals = {}
    fRatio = {}
    dfNum = 2
    dfDenom = 2*(n_side_comp*2) -1
    for cnd in cnds:
        fRatio[cnd] = {}
        fftVals[cnd] = {}
        fRatio[cnd]['F'] = []
        fRatio[cnd]['pval'] = []
        fftVals[cnd]['sigPow'] = []
        fftVals[cnd]['noisePow'] = []
        for c in range(len(comp_idx)):
            sideBands = get_noise_sidebands(comp_idx, n_side_comp, n_excluded_comp, ffts[cnd]['mag'])
            noisePow = mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][comp_idx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio

def get_F_ratios2(ffts, comp_idx, n_side_comp, n_excluded_comp, other_exclude):
    """
    Add excluded components
    """
    cnds = ffts.keys()
    fftVals = {}
    fRatio = {}
    dfNum = 2
    dfDenom = 2*(n_side_comp*2) -1
    for cnd in cnds:
        fRatio[cnd] = {}
        fftVals[cnd] = {}
        fRatio[cnd]['F'] = []
        fRatio[cnd]['pval'] = []
        fftVals[cnd]['sigPow'] = []
        fftVals[cnd]['noisePow'] = []
        for c in range(len(comp_idx)):
            sideBands = get_noise_sidebands2(comp_idx, n_side_comp, n_excluded_comp, ffts[cnd]['mag'], other_exclude)
            noisePow = numpy.mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][comp_idx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio


def get_noise_sidebands(components, n_comp_side, n_exclude_side, fft_array):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    #components: a list containing the indexes of the target components
    #nCompSide: number of components used for each side band
    #n_exclude_side: number of components adjacent to to the target components to exclude
    #fft_array: array containing the fft values
    idxProtect = []; idxProtect.extend(components);
    for i in range(n_exclude_side):
        idxProtect.extend(numpy.array(components) + (i+1))
        idxProtect.extend(numpy.array(components) - (i+1))
    #idxProtect = sorted(idxProtect)
    #print(idxProtect)

    noiseBands = []
    for i in range(len(components)):
        loSide = []
        hiSide = []
        counter = 1
        while len(hiSide) < n_comp_side:
            currIdx = components[i] + n_exclude_side + counter
            if currIdx not in idxProtect:
                hiSide.append(fft_array[currIdx])
            counter = counter + 1
        counter = 1
        while len(loSide) < n_comp_side:
            currIdx = components[i] - n_exclude_side - counter
            if currIdx not in idxProtect:
                loSide.append(fft_array[currIdx])
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        
    return noiseBands

def get_noise_sidebands2(components, n_comp_side, n_exclude_side, fft_array, other_exclude=None):
    """
    the 2 has the possibility to exclude extra components, useful for distortion products
    """
    #components: a list containing the indexes of the target components
    #nCompSide: number of components used for each side band
    #n_exclude_side: number of components adjacent to to the target components to exclude
    #fft_array: array containing the fft values
    idxProtect = []; idxProtect.extend(components);
    if other_exclude != None:
        idxProtect.extend(other_exclude)
    for i in range(n_exclude_side):
        idxProtect.extend(numpy.array(components) + (i+1))
        idxProtect.extend(numpy.array(components) - (i+1))
    #idxProtect = sorted(idxProtect)
    #print(idxProtect)

    noiseBands = []
    for i in range(len(components)):
        loSide = []
        hiSide = []
        counter = 1
        while len(hiSide) < n_comp_side:
            currIdx = components[i] + n_exclude_side + counter
            if currIdx not in idxProtect:
                hiSide.append(fft_array[currIdx])
            counter = counter + 1
        counter = 1
        while len(loSide) < n_comp_side:
            currIdx = components[i] - n_exclude_side - counter
            if currIdx not in idxProtect:
                loSide.append(fft_array[currIdx])
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        
    return noiseBands


def merge_triggers_cnt(trig_array, trig_list, new_trig):
    """
    Take one or more triggers in trig_list, and substitute them with new_trig

    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    
    trig_array[numpy.in1d(trig_array, trig_list)] = new_trig

    return 

def merge_triggers_event_table(event_table, trig_list, new_trig):
    """
    Substitute the event table triggers listed in trig_list
    with new_trig

    Parameters
    ----------
    event_table : dict of int arrays
        The event table
    trig_list : array of ints
        The list of triggers to substitute
    new_trig : int
        The new trigger used to substitute the triggers
        in trig_list
    Returns
    ----------

    Examples
    ----------
    """
    
    event_table['trigs'][numpy.in1d(event_table['trigs'], trig_list)] = new_trig
   
    return 

def read_biosig(file_name):
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
    HDR = biosig.sopen(file_name, 'r', HDR)
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
    for k in range(HDR.EVENT.N):
        codes.append(TYP[k] & (256-1))
        pos.append(int(POS[k]))
        

 
    # close file
    biosig.sclose(HDR)
    #
    # release allocated memory
    biosig.destructHDR(HDR)

    return data, array(codes), array(pos)

def remove_epochs(rec, to_remove):
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
        rec[code] = numpy.delete(rec[code], to_remove[code], axis=2)
    
    return 

def remove_spurious_triggers(event_table, sent_trigs, min_int, samp_rate):
    """
    Remove spurious trigger codes.

    Parameters
    ----------
    event_table :  dict with the following keys
       - trigs: array of ints
           The list of triggers in the EEG recording.
       - trigs_pos : array of ints
           The indexes of trigs in the EEG recording.
    sent_triggers : array of floats
        Array containing the list of triggers that were sent to the EEG recording equipment.
    min_int : float
        The minimum possible time interval between consecutive triggers in seconds
    samp_rate : int
        The sampling rate of the EEG recording


    Returns
    -------
    event_table :  dict with the following keys
       - trigs: array of ints
          List of valid triggers.
       - trigs_pos : array of ints
          The indexes of trigs in the EEG recording

    res_info: dict with the following keys:
       - len_matching: int
          Number of matching elements in the event table and sent_trigs
       - len_sent: int
          Length of sent_trigs
       - match : boolean
          True if a sequence matching the sent_trigs sequence is found in the event_table
    
    Examples
    --------
    >>> 
    ... 
    >>> 
    ... 
    >>> 
    """
    rec_trigs = event_table['trigs']
    rec_trigs_idx = event_table['trigs_idx']

    allowed_trigs = numpy.unique(sent_trigs)
    rec_trigs_idx = rec_trigs_idx[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs = rec_trigs[numpy.in1d(rec_trigs, allowed_trigs)]

    intervals_ok = False
    while intervals_ok == False:
        intervals = numpy.diff(rec_trigs_idx) / samp_rate
        intervals = numpy.insert(intervals, 0, min_int+1)
        if intervals[intervals < min_int].shape[0] == 0:
            intervals_ok = True
        else:
            idx_to_del = (numpy.where(intervals<min_int)[0][0])
            #print(rec_trigs_idx)
            rec_trigs = numpy.delete(rec_trigs, idx_to_del)
            rec_trigs_idx = numpy.delete(rec_trigs_idx, idx_to_del)
  
    ## trigs_to_discard = numpy.empty(0, dtype=numpy.int64); skip = 0
    ## for i in range(len(sent_trigs)):
    ##     if (i+skip) > (len(rec_trigs)-1):
    ##         break
    ##     if sent_trigs[i] != rec_trigs[i+skip]:
    ##         #print(rec_trigs_idx[i+skip])
    ##         trigs_to_discard = numpy.append(trigs_to_discard, i+skip)
    ##         alignment_found = False
    ##         while alignment_found == False:
    ##             skip = skip+1
    ##             if (i+skip) > (len(rec_trigs)-1):
    ##                 #print('Breaking while')
    ##                 break
    ##             if sent_trigs[i] != rec_trigs[i+skip]:# or intervals[i+skip] < min_int:
    ##                 trigs_to_discard = numpy.append(trigs_to_discard, i+skip)
    ##             else:
    ##                 alignment_found = True
    ## #print(trigs_to_discard)
    ## rec_trigs = numpy.delete(rec_trigs, trigs_to_discard)
    ## rec_trigs_idx = numpy.delete(rec_trigs_idx, trigs_to_discard)

    #rec_trigs = rec_trigs[0:len(sent_trigs)]
    #rec_trigs_idx = rec_trigs_idx[0:len(sent_trigs)]



            #print(idx_to_del)

   
   
    if numpy.array_equal(rec_trigs, sent_trigs) == True:
        match_found = True
    else:
        match_found = False

    event_table['trigs'] = rec_trigs
    event_table['trigs_idx'] = rec_trigs_idx

    res_info = {}
    res_info['match'] = match_found
    res_info['len_sent'] = len(sent_trigs)
    res_info['len_selected'] = len(rec_trigs)

    return res_info

def remove_spurious_triggers2(event_table, sent_trigs, min_trig_dur):
    rec_trigs = event_table['trigs']
    rec_trigs_dur = event_table['trigs_dur']
    rec_trigs_start = event_table['start_idx']
    rec_trigs_stop = event_table['stop_idx']
    
    allowed_trigs = numpy.unique(sent_trigs)
    rec_trigs_dur = rec_trigs_dur[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs_start = rec_trigs_start[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs_stop = rec_trigs_stop[numpy.in1d(rec_trigs, allowed_trigs)]
    rec_trigs = rec_trigs[numpy.in1d(rec_trigs, allowed_trigs)]

    rec_trigs = rec_trigs[rec_trigs_dur >= min_trig_dur]
    rec_trigs_start = rec_trigs_start[rec_trigs_dur >= min_trig_dur]
    rec_trigs_stop = rec_trigs_stop[rec_trigs_dur >= min_trig_dur]
    rec_trigs_dur = rec_trigs_dur[rec_trigs_dur >= min_trig_dur]

    if numpy.array_equal(rec_trigs, sent_trigs) == True:
        match_found = True
    else:
        match_found = False

    x = diff(rec_trigs_start)/2048
    print(x[x<1.375])
    print(min(x), max(x), mean(x))
    event_table['trigs'] = rec_trigs
    event_table['trigs_dur'] = rec_trigs_dur
    event_table['start_idx'] = rec_trigs_start
    event_table['stop_idx'] = rec_trigs_stop

    res_info = {}
    res_info['match'] = match_found
    res_info['len_sent'] = len(sent_trigs)
    res_info['len_found'] = len(rec_trigs)

    return res_info

def reref_cnt(rec, ref_channel, channels=None):
    """
    Rereference channels in a continuous recording.

    Parameters
    ----------
    rec : 
        Recording
    ref_channel: int
        The reference channel (indexing starts from zero).
    channels : list of ints
        List of channels to be rereferenced (indexing starts from zero).
  
    Returns
    -------
    rec : an array of floats with dimenions nChannels X nDataPoints
        
    Examples
    --------
    >>> reref_cnt(rec=dats, channels=[1, 2, 3], ref_channel=4)
    """

    if channels == None:
        nChannels = rec.shape[0]
        channels = list(range(nChannels))

    rec[channels,:] = rec[channels,:] - rec[ref_channel,:]

    return 


def save_F_ratios(file_name, subj, F_ratio, fft_values, cnds_trigs, cnds_labels, n_clean_by_block, n_raw_by_block):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    #cnds = list(F_ratio.keys())
    
    nRaw = {}
    nClean = {}
    for cnd in cnds_trigs:
        nRaw[cnd] = 0
        nClean[cnd] = 0
        for blk in range(len(n_clean_by_block)):
            nRaw[cnd] = nRaw[cnd] + n_raw_by_block[blk][cnd]
            nClean[cnd] = nClean[cnd] + n_clean_by_block[blk][cnd]
               
    subjVec = []
    compVec = []
    conditionVec = []
    nRawVec = []
    nCleanVec = []
    F_ratioVec = []
    sigPowVec = []
    noisePowVec = []
    pValVec = []
            
    for i in range(len(cnds_trigs)):
        thisN = len(F_ratio[cnds_trigs[i]]['F'])
        subjVec.extend(repeat(subj, thisN))
        conditionVec.extend(repeat(cnds_labels[i], thisN))
        compVec.extend(arange(thisN) + 1)
        sigPowVec.extend(fft_values[cnds_trigs[i]]['sigPow'][:])
        noisePowVec.extend(fft_values[cnds_trigs[i]]['noisePow'][:])
        pValVec.extend(F_ratio[cnds_trigs[i]]['pval'][:])
        F_ratioVec.extend(F_ratio[cnds_trigs[i]]['F'][:])
        nRawVec.extend(repeat(nRaw[cnds_trigs[i]], thisN))
        nCleanVec.extend(repeat(nClean[cnds_trigs[i]], thisN))
                
    datsFrame = DataFrame.from_items([('subj', subjVec), ('condition', conditionVec), ('comp', compVec), ('fRatio', F_ratioVec), ('pval', pValVec), ('sigPow', sigPowVec), ('noisePow', noisePowVec), ('nRaw', nRawVec), ('nClean', nCleanVec)])
    datsFrame['percRej'] = 100-((datsFrame['nClean'] / datsFrame['nRaw']) * 100)
    datsFrame.to_csv(file_name, sep=";")

def save_chained(din, d1, data_chan, data_string, ref_string):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    cnds = d1.keys()
    din[data_string+ref_string] = {}
    for cnd in cnds:
        din[data_string+ref_string][cnd] = copy.deepcopy(d1[cnd][data_chan,:])
    return din
    
    

def segment_cnt(rec, event_table, epoch_start, epoch_end, samp_rate, events_list=None):
    """
    Segment a continuous EEG recording into discrete event-related epochs.
    
    Parameters
    ----------
    rec: array of floats
        The EEG data.
    event_table : dict with the following keys
       - trigs : array of ints
           The list of triggers in the EEG recording.
       - trigs_pos : array of ints
           The indexes of trigs in the EEG recording.
    epoch_start : float
        The time at which the epoch starts relative to the trigger code, in seconds.
    epoch_end : float
        The time at which the epoch ends relative to the trigger code, in seconds.
    samp_rate : int
        The sampling rate of the EEG recording.
    events_list : list of ints
        The list of events for which epochs should be extracted.
        If no list is given epochs will be extracted for all the trigger
        codes present in the event table.
    
    Returns
    ----------
    segs : dict of 3D arrays
        The segmented recording. The dictionary has a key for each condition.
        The corresponding key value is a 3D array with dimensions
        n_channels x n_samples x n_segments
    n_segs : dict of ints
        The number of segments for each condition.
        
    Examples
    ----------
    >>>  segs, n_segs = eeg.segment_cnt(rec=dats, event_table=evt_tab, epoch_start=-0.2, epoch_end=0.8, samp_rate=512, events_list=['200', '201'])
    """
    if events_list == None:
        events_list = numpy.unique(trigs)

    trigs = event_table['trigs']
    trigs_pos = event_table['start_idx']
    epoch_start_sample = int(round(epoch_start*samp_rate))
    epoch_end_sample = int(round(epoch_end*samp_rate))

    nSamples = epoch_end_sample - epoch_start_sample
    segs = {}
    for i in range(len(events_list)):
        idx = trigs_pos[numpy.where(trigs == events_list[i])[0]]
        segs[str(events_list[i])] = numpy.zeros((rec.shape[0], nSamples, len(trigs[trigs==events_list[i]])))
        for j in range(len(idx)):
            thisStartPnt = (idx[j]+epoch_start_sample)
            #print(thisStartPnt)
            thisStopPnt = (idx[j]+epoch_end_sample)
            if thisStartPnt < 0 or thisStopPnt > rec.shape[1]:
                if thisStartPnt < 0:
                    print(idx[j], "Epoch starts before start of recording. Skipping")
                if thisStopPnt > rec.shape[1]:
                    print(idx[j], "Epoch ends after end of recording. Skipping")
            else:
                segs[str(events_list[i])][:,:,j] = rec[:, thisStartPnt:thisStopPnt]
    n_segs = {}
    for i in range(len(events_list)): #count
            n_segs[str(events_list[i])] = segs[str(events_list[i])].shape[2]

    return segs, n_segs



        


#Utility functions
#############
def next_pow_two(x):
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

def get_fft(sig, samp_rate, window, power_of_two):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if power_of_two == True:
        nfft = 2**next_pow_two(n)
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
    p = p / samp_rate  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    #p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (samp_rate / nfft);
    x = {'freqArray': freqArray, 'mag':p}
    return x

def get_spectrogram(sig, samp_freq, win_length, overlap, win_type, power_of_two):
    """
    winLength in seconds
    overlap in percent
    if the signal length is not a multiple of the window length it is trucated
    """
    winLengthPnt = floor(win_length * samp_freq)
    step = winLengthPnt - round(winLengthPnt * overlap / 100.)
    ind = arange(0, len(sig) - winLengthPnt, step)
    n = len(ind)

    x = get_spectrum(sig[ind[0]:ind[0]+winLengthPnt], samp_freq, win_type, power_of_two)
    freq_array = x['freq']; p = x['mag']

    power_matrix = zeros((len(freq_array), n))
    power_matrix[:,0] = p
    for i in range(1, n):
        x = get_spectrum(sig[ind[i]:ind[i]+winLengthPnt], samp_freq, win_type, power_of_two)
        freq_array = x['freq']; p = x['mag']
        power_matrix[:,i] = p

    timeInd = arange(0, len(sig), step)
    time_array = 1./samp_freq * (timeInd)
    x = {'freq': freq_array, 'time': time_array, 'mag': power_matrix}
    return x

def get_spectrum(sig, samp_rate, window, power_of_two):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if power_of_two == True:
        nfft = 2**next_pow_two(n)
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
    p = p / samp_rate  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freq_array = arange(0, nUniquePts, 1.0) * (samp_rate / nfft);
    x = {'freq': freq_array, 'mag':p}
    return x


def fir2_filt(f1, f2, f3, f4, snd, samp_rate, n_taps):
    """
    Filter signal with a fir2 filter.

    This function designs and applies a fir2 filter to a sound.
    The frequency response of the ideal filter will transition
    from 0 to 1 between 'f1' and 'f2', and from 1 to zero
    between 'f3' and 'f4'. The frequencies must be given in
    increasing order.

    Parameters
    ----------
    f1 : float
        Frequency in hertz of the point at which the transition
        for the low-frequency cutoff ends. 
    f2 : float
        Frequency in hertz of the point at which the transition
        for the low-frequency cutoff starts.
    f3 : float
        Frequency in hertz of the point at which the transition
        for the high-frequency cutoff starts.
    f4 : float
        Frequency in hertz of the point at which the transition
        for the high-frequency cutoff ends. 
    snd : array of floats
        The sound to be filtered.
    samp_rate : int
        Sampling frequency of 'snd'.

    Returns
    -------
    snd : 2-dimensional array of floats

    Notes
    -------
    If 'f1' and 'f2' are zero the filter will be lowpass.
    If 'f3' and 'f4' are equal to or greater than the nyquist
    frequency (samp_rate/2) the filter will be highpass.
    In the other cases the filter will be bandpass.

    The order of the filter (number of taps) is fixed at 256.
    This function uses internally 'scipy.signal.firwin2'.
       
    Examples
    --------
    >>> noise = broadbandNoise(spectrumLevel=40, duration=180, ramp=10,
    ...     channel='Both', samp_rate=48000, maxLevel=100)
    >>> lpNoise = fir2_filt(f1=0, f2=0, f3=1000, f4=1200, 
    ...     snd=noise, samp_rate=48000) #lowpass filter
    >>> hpNoise = fir2_filt(f1=0, f2=0, f3=24000, f4=26000, 
    ...     snd=noise, samp_rate=48000) #highpass filter
    >>> bpNoise = fir2_filt(f1=400, f2=600, f3=4000, f4=4400, 
    ...     snd=noise, samp_rate=48000) #bandpass filter
    """

    f1 = (f1 * 2) / samp_rate
    f2 = (f2 * 2) / samp_rate
    f3 = (f3 * 2) / samp_rate
    f4 = (f4 * 2) / samp_rate


    if f2 == 0: #low pass
        #print('lowpass')
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
        
    elif f3 < 1: #bandpass
        #print('bandpass')
        f = [0, f1, f2, ((f2+f3)/2), f3, f4, 1]
        m = [0, 0.00003, 1, 1, 1, 0.00003, 0]
        
    else:
        #print('highpass')
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
        
        
    b = firwin2 (n_taps,f,m);
    x = copy.copy(snd)
    x = convolve(snd, b, 1)
    #x[:, 1] = convolve(snd[:,1], b, 1)
    
    return x
