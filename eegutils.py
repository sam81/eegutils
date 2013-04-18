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

 Examples
 --------
 >>> 

"""
from __future__ import division
import copy, numpy
from numpy import abs, array, convolve, zeros
from scipy import signal
from scipy.signal import firwin2
from utils import*
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame
try:
    import biosig
except ImportError:
    pass
import ctypes

def reref_cnt(rec=None, channels=None, ref_channel=None):
    """
    Rereference channels in a continuous recording.

    Parameters
    ----------
    rec : 
        Recording
    channels : list of ints
        Channels to be rereferenced
    ref_channel: int
        The reference channel

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
    

def segment_cnt(rec=None, trigs=None, epochStart=None, epochEnd=None, eventsList=None, sampRate=None):
    if eventsList == None:
        eventsList = numpy.unique(trigs)

    epochStartSample = int(round(epochStart*sampRate))
    epochEndSample = int(round(epochEnd*sampRate))
    #print("Epoch start sample", epochStartSample)
    #print("Epoch end sample", epochEndSample)
    #add line to check start and end are within bounds
    segs = {}
    for i in range(len(eventsList)):
        idx = numpy.where(trigs == eventsList[i])[0]
        segs[str(eventsList[i])] = []
        for j in range(len(idx)):
            thisStartPnt = (idx[j]+epochStartSample)
            thisStopPnt = (idx[j]+epochEndSample)
            if thisStartPnt < 0 or thisStopPnt > rec.shape[1]:
                if thisStartPnt < 0:
                    print(idx[j], "Epoch starts before start of recording. Skipping")
                if thisStopPnt > rec.shape[1]:
                    print(idx[j], "Epoch ends after end of recording. Skipping")
            else:
                segs[str(eventsList[i])].append(rec[:, thisStartPnt:thisStopPnt])

    nSegs = {}
    for i in range(len(eventsList)): #count
            nSegs[str(eventsList[i])] = len(segs[str(eventsList[i])])
    return segs, nSegs

def segment_cnt_evt_tab(rec=None, trigs=None, trigs_pos=None, epochStart=None, epochEnd=None, eventsList=None, sampRate=None):
    if eventsList == None:
        eventsList = numpy.unique(trigs)

    epochStartSample = int(round(epochStart*sampRate))
    epochEndSample = int(round(epochEnd*sampRate))
    #print("Epoch start sample", epochStartSample)
    #print("Epoch end sample", epochEndSample)
    #add line to check start and end are within bounds
    segs = {}
    for i in range(len(eventsList)):
        idx = trigs_pos[numpy.where(trigs == eventsList[i])[0]]
        segs[str(eventsList[i])] = []
        for j in range(len(idx)):
            thisStartPnt = (idx[j]+epochStartSample)
            thisStopPnt = (idx[j]+epochEndSample)
            if thisStartPnt < 0 or thisStopPnt > rec.shape[1]:
                if thisStartPnt < 0:
                    print(idx[j], "Epoch starts before start of recording. Skipping")
                if thisStopPnt > rec.shape[1]:
                    print(idx[j], "Epoch ends after end of recording. Skipping")
            else:
                segs[str(eventsList[i])].append(rec[:, thisStartPnt:thisStopPnt])

    nSegs = {}
    for i in range(len(eventsList)): #count
            nSegs[str(eventsList[i])] = len(segs[str(eventsList[i])])
    return segs, nSegs

def merge_triggers_cnt(trig_array=None, trig_list=None, new_trig=None):
    """
    take one or more triggers in trig_list, and substitute them with new_trig
    """
    for trig in trig_list:
        trig_array[numpy.where(trig_array==trig)] = new_trig
    return trig_array

def baseline_correct(rec, bsStart, preDur, sampRate):
    """
    preDur:  duration of recording before experimental event
    bsDur: duration of the baseline, it cannot be greater than preDur
    """
    eventList = list(rec.keys())
    epochStartSample = int(round(preDur*sampRate))
    bsStartSample = int(epochStartSample - abs(round(bsStart*sampRate)))
   
    for i in range(len(eventList)): #for each event
        for j in range(len(rec[str(eventList[i])])): #for each epoch
            for k in range(rec[str(eventList[i])][j].shape[0]): #for each electrode
                thisBaseline = numpy.mean(rec[str(eventList[i])][j][k,bsStartSample:epochStartSample])
                rec[str(eventList[i])][j][k,:] = rec[str(eventList[i])][j][k,:] - thisBaseline
    return rec


def find_artefact_thresh(rec=None, thresh_lower=[-100], thresh_higher=[100], channels=None):
    eventList = list(rec.keys())
    segs_to_reject = {}
    for i in range(len(eventList)):
        segs_to_reject[str(eventList[i])] = [] #list to keep the indices of the epochs to delete
        for j in range(len(rec[str(eventList[i])])): #for each epoch
            for k in range(rec[str(eventList[i])][j].shape[0]): #for each channel
                if k in channels:
                    if (max(rec[str(eventList[i])][j][k,:]) > thresh_higher[channels.index(k)] or min(rec[str(eventList[i])][j][k,:]) < thresh_lower[channels.index(k)]) == True:
                        segs_to_reject[str(eventList[i])].append(j)
                
            
    for i in range(len(eventList)):
        segs_to_reject[str(eventList[i])] = numpy.unique(segs_to_reject[str(eventList[i])])

    return segs_to_reject


def remove_artefact(rec, to_remove):
    eventList = list(rec.keys())
    for code in eventList:
        currRem = list(to_remove[str(code)])
        currRem.reverse()
        if currRem != None:
            for item in currRem:
                rec[str(code)].pop(item)
    
    return rec

def get_average(rec=None):
    eventList = list(rec.keys())
    ave = {}
    nSegs = {}
    for code in eventList:
        ave[code] = numpy.zeros(rec[code][0].shape)
        for item in rec[code]:
            ave[code] = ave[code] + item
        nSegs[code] = len(rec[code])
        ave[code] = ave[code] / nSegs[code]
        
    return ave, nSegs

def average_averages(ave_list=None, nSegments=None):
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
        eegChained[currCode] = zeros((rec[currCode][0].shape[0], sweep_size))  #two-dimensional array of zeros
        fromeegChainedAve[currCode] = zeros((rec[currCode][0].shape[0], chunk_size))
        nReps[currCode] = zeros((n_chunks))
        p = 0
        k = 0
        while k < len(rec[currCode]):
            if p > (n_chunks-1):
                p = 0
            
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] + rec[currCode][k][:,startPnt:endPnt]
            nReps[currCode][p] = nReps[currCode][p] + 1
            fromeegChainedAve[currCode] = fromeegChainedAve[currCode] + rec[currCode][k][:,startPnt:endPnt]
            p = p+1 #p is the chunk counter
            k = k+1 #k is the epoch counter

    for i in range(len(eventList)):
        currCode = eventList[i]
        for p in range(n_chunks):
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] / nReps[currCode][p]
        fromeegChainedAve[currCode] = fromeegChainedAve[currCode] / sum(nReps[currCode])
    return eegChained#, fromeegChainedAve


def get_noise_sidebands(components, nCmpSide, nExcludeSide, fftArray):
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



def filter_segmented(rec, channels, samp_rate, filtertype, ntaps, cutoffs, transition_width):
    
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
        for i in range(len(rec[ev])):
            for j in range(rec[ev][0].shape[0]):
                if j in channels:
                    rec[ev][i][j,:] = convolve(rec[ev][i][j,:], b, 'same')
                    rec[ev][i][j,:] = convolve(rec[ev][i][j,:][::-1], b, 'same')[::-1]
    return(rec)
        
def filter_continuous(rec, channels, samp_rate, filtertype, ntaps, cutoffs, transition_width):
       
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


def read_biosig(fName):
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
        
def getFRatios(ffts, compIdx, nSideComp, nExcludedComp):
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
            noisePow = mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][compIdx[c]]
            thisF =  sigPow/ noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    return fftVals, fRatio

def saveFRatios(fName, subj, fRatio, fftVals, cnds_trigs, cndsLabels, nCleanByBlock, nRawByBlock):
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
    cnds = d1.keys()
    din[datastring+refstring] = {}
    for cnd in cnds:
        din[datastring+refstring][cnd] = copy.deepcopy(d1[cnd][data_chan,:])
    return din
    
## def combine_chained(d1, d2):
##     cnds = d1.keys()
##     cmb = {}
##     for cnd in cnds:
##         cmb[cnd] = (d1[cnd] + d2[cnd])/2
##     return cmb

def combine_chained(dList):
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
