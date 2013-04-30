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
This module contains functions to process electroencephalographic
recordings.
"""
from __future__ import division
import copy, numpy
from numpy import abs, arange, array, convolve, ceil, mean, repeat, zeros
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

def average_epochs(rec):
    """
    Average the epochs of a segmented recording.
    Parameters
    ----------
    rec : dict where each element is a 3D array with dimensions (nChannels x nSamples x nEpochs)
        Recording
    Returns
    ----------
    ave : dict where each element is a 2D array with dimensions (nChannels x nSamples)
    nSegs : dict of ints
        The number of epochs averaged for each condition
    Examples
    ----------
    """
    eventList = list(rec.keys())
    ave = {}
    nSegs = {}
    for code in eventList:
        nSegs[code] = rec[code].shape[2]
        ave[code] = numpy.mean(rec[code], axis=2)

    return ave, nSegs

def average_averages(ave_list, nSegments):
    """
    Perform a weighted average of a list of averages. The weight of
    each average in the list is determined by the number of segments
    from which it was obtained.
    
    Parameters
    ----------
    ave_list
    nSegments
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
            nSegsSum[event] = nSegsSum[event] + nSegments[i][event]

    for event in eventList:
        weightedAve[event] = numpy.zeros(ave_list[0][event].shape)
        for i in range(len(ave_list)):
           weightedAve[event] = weightedAve[event] + ave_list[i][event] * (nSegments[i][event]/nSegsSum[event])
    
    return weightedAve, nSegsSum

def baseline_correct(rec, bsStart, preDur, sampRate):
    """
    preDur:  duration of recording before experimental event
    bsDur: duration of the baseline, it cannot be greater than preDur
    """
    eventList = list(rec.keys())
    epochStartSample = int(round(preDur*sampRate))
    bsStartSample = int(epochStartSample - abs(round(bsStart*sampRate)))
   
    for i in range(len(eventList)): #for each event
        for j in range(rec[str(eventList[i])].shape[2]): #for each epoch
            for k in range(rec[str(eventList[i])].shape[0]): #for each electrode
                thisBaseline = numpy.mean(rec[str(eventList[i])][k,bsStartSample:epochStartSample,j])
                rec[str(eventList[i])][k,:,j] = rec[str(eventList[i])][k,:,j] - thisBaseline
    return rec
    
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

def combine_chained(dList):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    cnds = dList[0].keys()
    cmb = {}
    for cnd in cnds:
        for i in range(len(dList)):
            if i == 0:
                cmb[cnd] = dList[0][cnd]
            else:
                cmb[cnd] = cmb[cnd] + dList[i][cnd]
        cmb[cnd] = cmb[cnd] / len(dList)
            
    return cmb

def detrend(rec):
    nChannels = rec.shape[0]
    for i in range(nChannels):
        rec[i,:] = rec[i,:] - numpy.mean(rec[i,:])
    return rec
def detrend_segmentsed(rec):
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

def filter_segmented(rec, channels, samp_rate, filtertype, ntaps, cutoffs, transition_width):
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
   
    if filtertype == "lowpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filtertype == "highpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filtertype == "bandpass":
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
    b = firwin2 (ntaps,f,m);
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
    return(rec)
        
def filter_continuous(rec, channels, samp_rate, filtertype, ntaps, cutoffs, transition_width):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
       
    if filtertype == "lowpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f3, f4, 1]
        m = [1, 1, 0.00003, 0]
    elif filtertype == "highpass":
        f1 = cutoffs[0] * (1-transition_width)
        f2 = cutoffs[0]
        f1 = (f1*2) / samp_rate
        f2 = (f2*2) / samp_rate
        f = [0, f1, f2, 0.999999, 1] #high pass
        m = [0, 0.00003, 1, 1, 0]
    elif filtertype == "bandpass":
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
    b = firwin2 (ntaps,f,m);

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

def getFRatios(ffts, compIdx, nSideComp, nExcludedComp):
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
    dfDenom = 2*(nSideComp*2) -1
    for cnd in cnds:
        fRatio[cnd] = {}
        fftVals[cnd] = {}
        fRatio[cnd]['F'] = []
        fRatio[cnd]['pval'] = []
        fftVals[cnd]['sigPow'] = []
        fftVals[cnd]['noisePow'] = []
        for c in range(len(compIdx)):
            sideBands = get_noise_sidebands(compIdx, nSideComp, nExcludedComp, ffts[cnd]['mag'])
            noisePow = mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][compIdx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio

def getFRatios2(ffts, compIdx, nSideComp, nExcludedComp, other_exclude):
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
            sideBands = get_noise_sidebands2(compIdx, nSideComp, nExcludedComp, ffts[cnd]['mag'], other_exclude)
            noisePow = numpy.mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][compIdx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio


def get_noise_sidebands(components, nCmpSide, nExcludeSide, fftArray):
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
    #nExcludeSide: number of components adjacent to to the target components to exclude
    #fftArray: array containing the fft values
    idxProtect = []; idxProtect.extend(components);
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
        while len(hiSide) < nCmpSide:
            currIdx = components[i] + nExcludeSide + counter
            if currIdx not in idxProtect:
                hiSide.append(fftArray[currIdx])
            counter = counter + 1
        counter = 1
        while len(loSide) < nCmpSide:
            currIdx = components[i] - nExcludeSide - counter
            if currIdx not in idxProtect:
                loSide.append(fftArray[currIdx])
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        
    return noiseBands

def get_noise_sidebands2(components, nCmpSide, nExcludeSide, fftArray, other_exclude=None):
    """
    the 2 has the possibility to exclude extra components, useful for distortion products
    """
    #components: a list containing the indexes of the target components
    #nCompSide: number of components used for each side band
    #nExcludeSide: number of components adjacent to to the target components to exclude
    #fftArray: array containing the fft values
    idxProtect = []; idxProtect.extend(components);
    if other_exclude != None:
        idxProtect.extend(other_exclude)
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
        while len(hiSide) < nCmpSide:
            currIdx = components[i] + nExcludeSide + counter
            if currIdx not in idxProtect:
                hiSide.append(fftArray[currIdx])
            counter = counter + 1
        counter = 1
        while len(loSide) < nCmpSide:
            currIdx = components[i] - nExcludeSide - counter
            if currIdx not in idxProtect:
                loSide.append(fftArray[currIdx])
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        
    return noiseBands


def merge_triggers_cnt(trig_array, trig_list, new_trig):
    """
    take one or more triggers in trig_list, and substitute them with new_trig
    """
    for trig in trig_list:
        trig_array[numpy.where(trig_array==trig)] = new_trig
    return trig_array

def merge_triggers_event_table(event_table, trig_list, new_trig):
    """
    take one or more triggers in trig_list, and substitute them with new_trig
    """
    for trig in trig_list:
        event_table['trigs'][numpy.where(event_table['trigs']==trig)] = new_trig
    return event_table

def read_biosig(fName):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    HDR = biosig.constructHDR(0,0)
    HDR = biosig.sopen(fName, 'r', HDR)
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

def remove_artefact(rec, to_remove):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    eventList = list(rec.keys())
    for code in eventList:
        rec[code] = numpy.delete(rec[code], to_remove[code], axis=2)
    
    return rec

def remove_spurious_triggers(event_table, sent_trigs):
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
    
    trigs_to_discard = numpy.empty(0, dtype=numpy.int64); skip = 0
    for i in range(len(sent_trigs)):
        if (i+skip) > (len(rec_trigs)-1):
            break
        if sent_trigs[i] != rec_trigs[i+skip]:
            trigs_to_discard = numpy.append(trigs_to_discard, i+skip)
            alignment_found = False
            while alignment_found == False:
                skip = skip+1
                if (i+skip) > (len(rec_trigs)-1):
                    #print('Breaking while')
                    break
                if(sent_trigs[i] != rec_trigs[i+skip]):
                    trigs_to_discard = numpy.append(trigs_to_discard, i+skip)
                else:
                    alignment_found = True
                    
    rec_trigs = numpy.delete(rec_trigs, trigs_to_discard)
    rec_trigs_idx = numpy.delete(rec_trigs_idx, trigs_to_discard)
    
    rec_trigs = rec_trigs[0:len(sent_trigs)]
    rec_trigs_idx = rec_trigs_idx[0:len(sent_trigs)]
    if len(numpy.where((rec_trigs == sent_trigs) == False)[0]) > 0:
        match_found = False
    else:
        match_found = True

    event_table['trigs'] = rec_trigs
    event_table['trigs_idx'] = rec_trigs_idx

    res_info = {}
    res_info['match'] = match_found
    res_info['len_sent'] = len(sent_trigs)
    res_info['len_matching'] = len(rec_trigs)

    return event_table, res_info



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

    nChannels = rec.shape[0]
    if channels == None:
        channels = list(range(nChannels))
    for i in range(nChannels):
        if i in channels and i != ref_channel:
            rec[i,:] = rec[i,:] - rec[ref_channel,:]
    rec[ref_channel,:] = 0
    return rec


def saveFRatios(fName, subj, fRatio, fftVals, cnds_trigs, cndsLabels, nCleanByBlock, nRawByBlock):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    #cnds = list(fRatio.keys())
    
    nRaw = {}
    nClean = {}
    for cnd in cnds_trigs:
        nRaw[cnd] = 0
        nClean[cnd] = 0
        for blk in range(len(nCleanByBlock)):
            nRaw[cnd] = nRaw[cnd] + nRawByBlock[blk][cnd]
            nClean[cnd] = nClean[cnd] + nCleanByBlock[blk][cnd]
               
    subjVec = []
    compVec = []
    conditionVec = []
    nRawVec = []
    nCleanVec = []
    fRatioVec = []
    sigPowVec = []
    noisePowVec = []
    pValVec = []
            
    for i in range(len(cnds_trigs)):
        thisN = len(fRatio[cnds_trigs[i]]['F'])
        subjVec.extend(repeat(subj, thisN))
        conditionVec.extend(repeat(cndsLabels[i], thisN))
        compVec.extend(arange(thisN) + 1)
        sigPowVec.extend(fftVals[cnds_trigs[i]]['sigPow'][:])
        noisePowVec.extend(fftVals[cnds_trigs[i]]['noisePow'][:])
        pValVec.extend(fRatio[cnds_trigs[i]]['pval'][:])
        fRatioVec.extend(fRatio[cnds_trigs[i]]['F'][:])
        nRawVec.extend(repeat(nRaw[cnds_trigs[i]], thisN))
        nCleanVec.extend(repeat(nClean[cnds_trigs[i]], thisN))
                
    datsFrame = DataFrame.from_items([('subj', subjVec), ('condition', conditionVec), ('comp', compVec), ('fRatio', fRatioVec), ('pval', pValVec), ('sigPow', sigPowVec), ('noisePow', noisePowVec), ('nRaw', nRawVec), ('nClean', nCleanVec)])
    datsFrame['percRej'] = 100-((datsFrame['nClean'] / datsFrame['nRaw']) * 100)
    datsFrame.to_csv(fName, sep=";")

def save_chained(din, d1, data_chan, datastring, refstring):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    cnds = d1.keys()
    din[datastring+refstring] = {}
    for cnd in cnds:
        din[datastring+refstring][cnd] = copy.deepcopy(d1[cnd][data_chan,:])
    return din
    
    

def segment_cnt(rec, event_table, epochStart, epochEnd, sampRate, eventsList=None):
    """
    Segment a continuous EEG recording into discrete event-related epochs.
    
    Parameters
    ----------
    rec: array of floats
        The EEG data.
    event_table :  dict with the following keys
       - trigs: array of ints
           The list of triggers in the EEG recording.
       - trigs_pos : array of ints
           The indexes of trigs in the EEG recording.
    epochStart : float
        The time at which the epoch starts relative to the trigger code, in seconds.
    epochEnd : float
        The time at which the epoch ends relative to the trigger code, in seconds.
    sampRate : int
        The sampling rate of the EEG recording.
    eventsList : list of ints
        The list of events for which epochs should be extracted.
        If no list is given epochs will be extracted for all the trigger
        codes present in the event table.
    
    Returns
    ----------
    segs : a dict with a list of epochs for each 
    
    Examples
    ----------
    """
    if eventsList == None:
        eventsList = numpy.unique(trigs)

    trigs = event_table['trigs']
    trigs_pos = event_table['trigs_idx']
    epochStartSample = int(round(epochStart*sampRate))
    epochEndSample = int(round(epochEnd*sampRate))

    nSamples = epochEndSample - epochStartSample
    segs = {}
    for i in range(len(eventsList)):
        idx = trigs_pos[numpy.where(trigs == eventsList[i])[0]]
        segs[str(eventsList[i])] = numpy.zeros((rec.shape[0], nSamples, len(trigs[trigs==eventsList[i]])))
        for j in range(len(idx)):
            thisStartPnt = (idx[j]+epochStartSample)
            thisStopPnt = (idx[j]+epochEndSample)
            if thisStartPnt < 0 or thisStopPnt > rec.shape[1]:
                if thisStartPnt < 0:
                    print(idx[j], "Epoch starts before start of recording. Skipping")
                if thisStopPnt > rec.shape[1]:
                    print(idx[j], "Epoch ends after end of recording. Skipping")
            else:
                segs[str(eventsList[i])][:,:,j] = rec[:, thisStartPnt:thisStopPnt]
    nSegs = {}
    for i in range(len(eventsList)): #count
            nSegs[str(eventsList[i])] = segs[str(eventsList[i])].shape[2]

    return segs, nSegs



        








#Utility functions
#############
def nextpow2(x):
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

def getFFT(sig, sampFreq, window, poweroftwo):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if poweroftwo == True:
        nfft = 2**nextpow2(n)
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
    p = p / sampFreq  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    #p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / nfft);
    x = {'freqArray': freqArray, 'mag':p}
    return x


def getSpectrum(sig, sampFreq, window, poweroftwo):
    """
    
    Parameters
    ----------

    Returns
    ----------

    Examples
    ----------
    """
    n = len(sig)
    if poweroftwo == True:
        nfft = 2**nextpow2(n)
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
    p = p / sampFreq  # scale by the number of points so that
    # the magnitude does not depend on the length 
    # of the signal or on its sampling frequency  
    p = p**2  # square it to get the power 

    # multiply by two (see technical document for details)
    # odd nfft excludes Nyquist point
    if nfft % 2 > 0: # we've got odd number of points fft
         p[1:len(p)] = p[1:len(p)] * 2
    else:
         p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / nfft);
    x = {'freqArray': freqArray, 'mag':p}
    return x


def fir2Filt(f1, f2, f3, f4, snd, fs, ntaps):
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
    fs : int
        Sampling frequency of 'snd'.

    Returns
    -------
    snd : 2-dimensional array of floats

    Notes
    -------
    If 'f1' and 'f2' are zero the filter will be lowpass.
    If 'f3' and 'f4' are equal to or greater than the nyquist
    frequency (fs/2) the filter will be highpass.
    In the other cases the filter will be bandpass.

    The order of the filter (number of taps) is fixed at 256.
    This function uses internally 'scipy.signal.firwin2'.
       
    Examples
    --------
    >>> noise = broadbandNoise(spectrumLevel=40, duration=180, ramp=10,
    ...     channel='Both', fs=48000, maxLevel=100)
    >>> lpNoise = fir2Filt(f1=0, f2=0, f3=1000, f4=1200, 
    ...     snd=noise, fs=48000) #lowpass filter
    >>> hpNoise = fir2Filt(f1=0, f2=0, f3=24000, f4=26000, 
    ...     snd=noise, fs=48000) #highpass filter
    >>> bpNoise = fir2Filt(f1=400, f2=600, f3=4000, f4=4400, 
    ...     snd=noise, fs=48000) #bandpass filter
    """

    f1 = (f1 * 2) / fs
    f2 = (f2 * 2) / fs
    f3 = (f3 * 2) / fs
    f4 = (f4 * 2) / fs


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
        
        
    b = firwin2 (ntaps,f,m);
    x = copy.copy(snd)
    x = convolve(snd, b, 1)
    #x[:, 1] = convolve(snd[:,1], b, 1)
    
    return x
