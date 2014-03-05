# -*- coding: utf-8 -*-
#   Copyright (C) 2012-2014 Samuele Carcagno <sam.carcagno@gmail.com>
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
This module contains functions to extract and process event related
potentials (ERPs) from electroencephalographic (EEG) recordings.
"""

from __future__ import division
import copy, numpy
from numpy import abs, append, arange, array, array_equal, convolve, ceil, diff, floor, log2, log10, mean, repeat, where, zeros
from numpy.fft import fft
from scipy import signal
from scipy.signal import bartlett, blackman, fftconvolve, firwin2, hamming, hanning
import matplotlib.pyplot as plt
import scipy.stats
from pandas import DataFrame
import numpy as np
try:
    import biosig
except ImportError:
    pass
import ctypes

__version__ = "0.0.4"

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
    rec : dict of 3D arrays
        The segmented recording.
    baselineStart : float
        Start time of the baseline window relative to the event onset, in seconds.
        The absolute value of baselineStart cannot be greater than preDur.
        In practice baselineStart allows you to define a baseline window shorter
        than the time window before the experimental event (preDur).
    preDur : float
        Duration of recording epoch before the experimental event, in seconds.
    sampRate : int
        The samplig rate of the EEG recording.

    Examples
    ----------
    >>> #baseline window has the same duration of preDur
    >>> baseline_correct(rec=rec, baselineStart=-0.2, preDur=0.2, sampRate=512)
    >>> #now with a baseline shorter than preDur
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
    
def chainSegments(rec, nChunks, sampRate, start, end, baselineDur=0, window=None):
    """
    Take a dictionary containing in each key a list of segments, and chain these segments
    into chunks of length `nChunks`. `baselineDur` is for determining what is the zero point.
    `start` and `end` are given with reference to the zero point.
    This chaining technique is used to increase the spectral resolution of FFT analyses
    of auditory steady-state responses.

    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented recordings for each experimental condition.
    nChunks : int
        The number of segments to chain together for each chunk.
    sampRate : int
        The EEG recording sampling rate.
    start : float
        Start time of the epoch segments to be chained, in seconds.
    end : float
        End time of the epoch segments to be chained, in seconds.
    baselineDur : float
        Duration of the baseline, in seconds.

    Returns
    ----------
    eegChained : dict of 2D arrays
        The chained recordings for each experimental condition.

    Examples
    ----------
    >>> chainSegments(rec, nChunks=20, sampRate=2048, start=0, end=0.5, baselineDur=0.1)
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
    n = chunk_size
    if window == 'hamming':
        w = hamming(n)
    elif window == 'hanning':
        w = hanning(n)
    elif window == 'blackman':
        w = blackman(n)
    elif window == 'bartlett':
        w = bartlett(n)
    elif window == 'tukey':
        w = tukeywin(n)
    for i in range(len(eventList)):
        currCode = eventList[i]
        for p in range(nChunks):
            idxChunkStart = p*chunk_size
            idxChunkEnd = idxChunkStart + chunk_size
            if window == None:
                eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd] / nReps[currCode][p]
            else:
                eegChained[currCode][:,idxChunkStart:idxChunkEnd] = eegChained[currCode][:,idxChunkStart:idxChunkEnd]*w / nReps[currCode][p]
        fromeegChainedAve[currCode] = fromeegChainedAve[currCode] / sum(nReps[currCode])
    return eegChained



def detrendEEG(rec):
    """
    Remove the mean value from each channel of an EEG recording.

    Parameters
    ----------
    rec : dict of 2D arrays
        The EEG recording.

    Examples
    ----------
    >>> detrend(rec)
    
    """
    nChannels = rec.shape[0]
    for i in range(nChannels):
        rec[i,:] = rec[i,:] - numpy.mean(rec[i,:], dtype=rec.dtype)
    return rec

def detrendSegmented(rec):
    """
    Remove the mean value from each channel of an EEG recording.

    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented EEG recording.

    Examples
    ----------
    >>> detrendSegmented(rec)
    
    """
    eventList = list(rec.keys())
    for ev in eventList:
        for i in range(len(rec[ev])):
            for j in range(rec[ev][0].shape[0]):
                rec[ev][i][j,:] = rec[ev][i][j,:] - numpy.mean(rec[ev][i][j,:])
    return(rec)


def extractEventTable(trigChan, sampRate):
    """
    Extract the event table from the EEG channel containing the trigger codes.

    Parameters
    ----------
    trigChan : array
        The trigger channel.
    sampRate : int
        The EEG recording sampling rate.

    Returns
    ----------
    eventTable : a dictionary with the following keys
        - code : array of ints
            The trigger codes.
        - idx : array of ints
            The indexes of the trigger codes.
        - dur : array of floats
            The duration of the triggers, in seconds.
          
    Examples
    ----------
    >>> evtTab = extractEventTable(trigChan, 2048)
    
    """
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

def getFilterFreqResp(sampRate, filterType, nTaps, cutoffs, transitionWidth, plotResp=False):
    """
    Get the frequency response of a eegutils filter.

    Parameters
    ----------
    sampRate : int
        The EEG recording sampling rate
    filterType : string {lowpass, highpass, bandpass}
        The filter type.
    nTaps : int
        The number of filter taps.
    cutoffs : array of floats
        The filter cutoffs. If 'filterType' is 'lowpass' or 'highpass'
        the 'cutoffs' array should contain a single value. If 'filterType'
        is bandpass the 'cutoffs' array should contain the lower and
        the upper cutoffs in increasing order.
    transitionWidth : float
        The width of the filter transition region, normalized between 0-1.
        For a lower cutoff the nominal transition region will go from
        `(1-transitionWidth)*cutoff` to `cutoff`. For a higher cutoff
        the nominal transition region will go from cutoff to
        `(1+transitionWidth)*cutoff`.
    plotResp : bool
        Whether to plot the frequency response.

    Returns
    ----------
    freq : array of floats
        The frequency axis.
    mag : array of floats
        The frequency response of the filter. This is an array
        of complex numbers, to get the real part use `abs(mag)`.

    Examples
    ----------
    >>>  f, m = getFilterFreqResp(2048, 'highpass', 512, [30], 0.2)
    """
    b = getFilterCoefficients(sampRate, filterType, nTaps, cutoffs, transitionWidth)
    freq,mag = signal.freqz(b,1)
    freq = (freq/max(freq))*(sampRate/2)

    if plotResp == True:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot(freq, 20*log10(abs(mag)))
        axes.set_xlabel('Frequency (Hz)')
        axes.set_ylabel('Level (dB)')
        plt.grid()
        axes.set_title("# taps: " + str(nTaps) + " - trans. width: " + str(transitionWidth))
        plt.show()

    return freq, mag
    
def getFilterCoefficients(sampRate, filterType, nTaps, cutoffs, transitionWidth):
    """
    Get the coefficients of a FIR filter. This function is used internally by eegutils.

    Parameters
    ----------
    sampRate : int
        The EEG recording sampling rate.
    filterType : str {'lowpass', 'highpass', 'bandpass'}
        The filter type.
    nTaps : int
        The number of filter taps.
    cutoffs : array of floats
        The filter cutoffs. If 'filterType' is 'lowpass' or 'highpass'
        the 'cutoffs' array should contain a single value. If 'filterType'
        is bandpass the 'cutoffs' array should contain the lower and
        the upper cutoffs in increasing order.
    transitionWidth : float
        The width of the filter transition region, normalized between 0-1.
        For a lower cutoff the nominal transition region will go from
        `(1-transitionWidth)*cutoff` to `cutoff`. For a higher cutoff
        the nominal transition region will go from cutoff to
        `(1+transitionWidth)*cutoff`.

    Returns
    ---------
    filterCoeff : array of floats
        The filter coefficients. 
        
    Examples
    ----------
    >>> getFilterCoefficients(sampRate=2048, filterType='highpass', nTaps=512, cutoffs=[30], transitionWidth=0.2)

    """
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
    filterCoeff = firwin2 (nTaps,f,m);
    
    return filterCoeff

def filterSegmented(rec, channels, sampRate, filterType, nTaps, cutoffs, transitionWidth):
    """
    Filter a segmented recording.
    
    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented EEG recording.
    channels : array of ints
        The list of channels that should be filtered.
    sampRate : int
        The EEG recording sampling rate.
    filterType : str {'lowpass', 'highpass', 'bandpass'}
        The filter type.
    nTaps : int
        The number of filter taps.
    cutoffs : array of floats
        The filter cutoffs. If 'filterType' is 'lowpass' or 'highpass'
        the 'cutoffs' array should contain a single value. If 'filterType'
        is bandpass the 'cutoffs' array should contain the lower and
        the upper cutoffs in increasing order.
    transitionWidth : float
        The width of the filter transition region, normalized between 0-1.
        For a lower cutoff the nominal transition region will go from
        `(1-transitionWidth)*cutoff` to `cutoff`. For a higher cutoff
        the nominal transition region will go from cutoff to
        `(1+transitionWidth)*cutoff`.
        
    Examples
    ----------
    >>> filterSegmented(rec=rec, channels=[0,1,2,3], sampRate=2048, filterType='highpass', nTaps=512, cutoffs=[30], transitionWidth=0.2)
    
    """
    
    eventList = list(rec.keys())

    nChannels = rec[eventList[0]][0].shape[0]
    if channels == None or len(channels) == 0:
        channels = list(range(nChannels))
   

    b = getFilterCoefficients(sampRate, filterType, nTaps, cutoffs, transitionWidth)
    b = b.astype(rec[eventList[0]].dtype)
    for ev in eventList:
        for i in range(rec[ev].shape[2]): #for each epoch
            for j in range(rec[ev].shape[0]): #for each channel
                if j in channels:
                    rec[ev][j,:,i] = fftconvolve(rec[ev][j,:,i], b, 'same')
                    rec[ev][j,:,i] = fftconvolve(rec[ev][j,:,i][::-1], b, 'same')[::-1]
    return(rec)
        
def filterContinuous(rec, channels, sampRate, filterType, nTaps, cutoffs, transitionWidth):
    """
    Filter a continuous recording.
    
    Parameters
    ----------
    rec : 2D array
        The nChannelsXnSamples array with the EEG recording.
    channels : array of ints
        The list of channels that should be filtered.
    sampRate : int
        The EEG recording sampling rate.
    filterType : str {'lowpass', 'highpass', 'bandpass'}
        The filter type.
    nTaps : int
        The number of filter taps.
    cutoffs : array of floats
        The filter cutoffs. If 'filterType' is 'lowpass' or 'highpass'
        the 'cutoffs' array should contain a single value. If 'filterType'
        is bandpass the 'cutoffs' array should contain the lower and
        the upper cutoffs in increasing order.
    transitionWidth : float
        The width of the filter transition region, normalized between 0-1.
        For a lower cutoff the nominal transition region will go from
        `(1-transitionWidth)*cutoff` to `cutoff`. For a higher cutoff
        the nominal transition region will go from cutoff to
        `(1+transitionWidth)*cutoff`.
        
    Examples
    ----------
    >>> filterContinuous(rec=rec, channels=[0,1,2,3], sampRate=2048, filterType='highpass', nTaps=512, cutoffs=[30], transitionWidth=0.2)
    
    """
       
    b = getFilterCoefficients(sampRate, filterType, nTaps, cutoffs, transitionWidth)
    b = b.astype(rec.dtype)
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
        The segmented recording.
    thresh : array of floats
        The threshold value for each channel listed in `channels`.
    channels = array or list of ints
        The indexes of the channels to check for artefacts.
        
    Returns
    ----------
    segsToReject : array of ints
        The indexes of the epochs exceeding the threshold.
        
    Examples
    ----------
    >>> toRemove = eeg.findArtefactThresh(rec=segs, thresh=[100,60,100], channels=[0,1,2])
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


## def getFRatios(ffts, compIdx, nSideComp, nExcludedComp, otherExclude):
##     """
##     Compute signal to noise ratio (SNR) of one or more signals from a fast
##     fourier transform (FFT) and test the SNR significance using an F-test.

##     Parameters
##     ----------
##     ffts : dict 
##         The ffts for each experimental condition. The ffts should be in the same
##         format as returned by the :func:`getSpectrum` function, i.e. a dictionary with
##         `freq` and `mag` keys.
##     compIdx : array of ints
##         The positional indexes of the signal components in the `fft['freq']` array.
##     nSideComp : int
##         The number of components adjacent to each side of the signal components
##         from which to estimate the noise power. `nSideComp` above and `nSideComp`
##         below each signal component will be used to estimate noise power around
##         each signal. In other words, the noise power around each signal component will
##         be estimated from `2*nSideComp` components.
##     nExcludedComp: int
##         To avoid that spectral leaks from the signal component affect the noise power
##         estimation, the `nExcludedComp` closest to the signal component will not be used
##         for estimating noise power.
##     otherExclude : array of ints
##         The indexes of other components to exclude from the computation of the noise power.
##         This may be useful to exclude components corresponding to distortion products
##         generated by the signal.

##     Returns
##     ----------
##     fftVals : dict
##         The signal and noise power for each component and experimental condition.
##         Each key of `fftVals` corresponds to an experimental condition. For each
##         experimental condition there is a dictionary with keys `noisePow` and `sigPow`
##         that list the noise and signal power for each component given in `compIdx`.
##     fRatio :
##         The F and corresponding p-value for each component and experimental condition.
##         Each key of `fRatio` corresponds to an experimental condition. For each
##         experimental condition there is a dictionary with keys `F` and `pval`
##         that list the F and p value for each component given in `compIdx`.
        
##     Examples
##     ----------
##     >>> getFRatios(ffts=ffts, compIdx=[30, 75], nSideComp=30, nExcludedComp=1, otherExclude=[25, 68])
##     """
##     cnds = ffts.keys()
##     fftVals = {}
##     fRatio = {}
##     dfNum = 2
##     dfDenom = 2*(nSideComp*2) -1
##     for cnd in cnds:
##         fRatio[cnd] = {}
##         fftVals[cnd] = {}
##         fRatio[cnd]['F'] = []
##         fRatio[cnd]['pval'] = []
##         fftVals[cnd]['sigPow'] = []
##         fftVals[cnd]['noisePow'] = []
##         sideBands, sideBandsIdx = getNoiseSidebands(compIdx, nSideComp, nExcludedComp, ffts[cnd]['mag'], otherExclude)
##         for c in range(len(compIdx)):
##             noisePow = numpy.mean(sideBands[c])
##             sigPow = ffts[cnd]['mag'][compIdx[c]]
##             thisF =  sigPow / noisePow
##             fftVals[cnd]['sigPow'].append(sigPow)
##             fftVals[cnd]['noisePow'].append(noisePow)
##             fRatio[cnd]['F'].append(thisF)
##             fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
##     return fftVals, fRatio

## def getNoiseSidebands(components, nCompSide, nExcludeSide, FFTArray, otherExclude=None):
##     """
##     Compute 

##     Parameters
##     ----------
##     components : list of ints
##         The indexes of the signal components.
##     nCompSide : int
##         The number of components adjacent to each side of the signal components
##         from which to estimate the noise power. `nSideComp` above and `nSideComp`
##         below each signal component will be used to estimate noise power around
##         each signal. In other words, the noise power around each signal component will
##         be estimated from `2*nSideComp` components.
##     nExcludeSide : int
##         To avoid that spectral leaks from the signal component affect the noise power
##         estimation, the `nExcludeSide` components closest to the signal component will not be used
##         for estimating noise power.
##     FFTArray: int
##         The array containing the fft magnitude values.
##     otherExclude : array of ints
##         The indexes of other components to exclude from the computation of the noise power.
##         This may be useful to exclude components corresponding to distortion products
##         generated by the signal.

##     Returns
##     ----------
##     noiseBands : list
##         A list a sub-list for each component specified in `components`. Each sublist
##         contains the fft magnitude values of the noise side bands.
        
##     Examples
##     ----------
##     >>> getNoiseSidebands(compIdx=[30, 75], nSideComp=30, nExcludedComp=2, FFTArray=fftVals, otherExclude=[25,68])
##     """
    
##     idxProtect = []; idxProtect.extend(components);
##     if otherExclude != None:
##         idxProtect.extend(otherExclude)
##     for i in range(nExcludeSide):
##         idxProtect.extend(numpy.array(components) + (i+1))
##         idxProtect.extend(numpy.array(components) - (i+1))

##     noiseBands = []; noiseBandsIdx = []
##     for i in range(len(components)):
##         loSide = []; loSideIdx = []
##         hiSide = []; hiSideIdx = []
##         counter = 1
##         while len(hiSide) < nCompSide:
##             currIdx = components[i] + nExcludeSide + counter
##             if currIdx not in idxProtect:
##                 hiSide.append(FFTArray[currIdx])
##                 hiSideIdx.append(currIdx)
##             counter = counter + 1
            
##         counter = 1
##         while len(loSide) < nCompSide:
##             currIdx = components[i] - nExcludeSide - counter
##             if currIdx not in idxProtect:
##                 loSide.append(FFTArray[currIdx])
##                 loSideIdx.append(currIdx)
##             counter = counter + 1
##         noiseBands.append(loSide+hiSide)
##         noiseBandsIdx.append(loSideIdx+hiSideIdx)
##     return noiseBands, noiseBandsIdx

def getFRatios(ffts, freqs, nSideComp, nExcludedComp, otherExclude):
    """
    Compute signal to noise ratio (SNR) of one or more signals from a fast
    fourier transform (FFT) and test the SNR significance using an F-test.

    Parameters
    ----------
    ffts : dict 
        The ffts for each experimental condition. The ffts should be in the same
        format as returned by the :func:`getSpectrum` function, i.e. a dictionary with
        `freq` and `mag` keys.
    freqs : array of floats
        The frequencies of the signals.
    nSideComp : int
        The number of components adjacent to each side of the signal components
        from which to estimate the noise power. `nSideComp` above and `nSideComp`
        below each signal will be used for each noise-power estimate. In other words,
        the noise power around each signal component will be estimated from `2*nSideComp`
        components.
    nExcludedComp: int
        To avoid that spectral leaks from the signal affect the noise-power estimate,
        the `nExcludedComp` components just above and the `nExcludeComp` components just
        below the signal will not be used for estimating noise power.
    otherExclude : array of ints
        The frequencies of other components to exclude from the computation of the noise power.
        This may be useful to exclude components corresponding to distortion products
        generated by the signal. The `nExcludedComp` components just above and the
        `nExcludeComp` components just below each component in `otherExclude` will
        also be excluded.

    Returns
    ----------
    res : dict with the following keys
       - fftVals : dict
          The signal and noise power for each component and experimental condition.
          Each key of `fftVals` corresponds to an experimental condition. For each
          experimental condition there is a dictionary with keys `noisePow` and `sigPow`
          that list the noise and signal power for each component given in `freqs`.
       - fRatio :
          The F and corresponding p-value for each component and experimental condition.
          Each key of `fRatio` corresponds to an experimental condition. For each
          experimental condition there is a dictionary with keys `F` and `pval`
          that list the F and p value for each component given in `freqs`.
       - compIdx : list
          The indexes of the signal frequencies in the FFT array.
       - sideBandsIdx : list
          The indexes of the noise side bands in the FFT array.
          A separate sub-list is returned for each component specified in `freqs`. 
       - excludedIdx : list
          The indexes of the components excluded from the noise side bands.
       - minSideFreq : list
          For each signal, the lowest frequency of the noise bands.
       - maxSideFreq : list
          For each signal, the highest frequency of the noise bands.
        
    Examples
    ----------
    >>> getFRatios(ffts=ffts, freqs=[30, 75], nSideComp=30, nExcludedComp=1, otherExclude=[25, 68])
    """
  
    cnds = list(ffts.keys())
    compIdx = []
    for i in range(len(freqs)):
        compIdx.append(where(abs(ffts[cnds[0]]['freq'] - freqs[i]) == min(abs(ffts[cnds[0]]['freq'] - freqs[i])))[0][0])
 
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
        sideBands, sideBandsIdx, idxProtect = getNoiseSidebands(freqs, nSideComp, nExcludedComp, ffts[cnd], otherExclude)
        #print(freqs)
        for c in range(len(compIdx)):
            noisePow = numpy.mean(sideBands[c])
            sigPow = ffts[cnd]['mag'][compIdx[c]]
            thisF =  sigPow / noisePow
            fftVals[cnd]['sigPow'].append(sigPow)
            fftVals[cnd]['noisePow'].append(noisePow)
            fRatio[cnd]['F'].append(thisF)
            fRatio[cnd]['pval'].append(scipy.stats.f.pdf(thisF, dfNum, dfDenom))
    minSideFreq = []
    maxSideFreq = []
    for c in range(len(compIdx)):
        minSideFreq.append(ffts[cnds[0]]['freq'][min(sideBandsIdx[c])])
        maxSideFreq.append(ffts[cnds[0]]['freq'][max(sideBandsIdx[c])])
    res = {'fftVals': fftVals, 'fRatio': fRatio, 'compIdx': compIdx, 'sideBandsIdx':sideBandsIdx, 'excludedIdx': idxProtect, 'minSideFreq': minSideFreq, 'maxSideFreq': maxSideFreq}
    return res

def getNoiseSidebands(componentsFreq, nCompSide, nExcludedComp, fftDict, otherExclude=None):
    """
    Given one or more signal frequencies, get, for each signal frequency, the 
    power in frequency bins adjacent to the signal frequency. The results can be used to
    estimate *local* noise in signal-to-noise-ratio computations.

    Parameters
    ----------
    componentsFreq : list of floats
        The frequencies of the signal components.
    nCompSide : int
        The number of components adjacent to each side of the signal components
        from which to estimate the noise power. `nSideComp` above and `nSideComp`
        below each signal will be used for each noise-power estimate. In other words,
        the noise power around each signal component will be estimated from `2*nSideComp`
        components.
    nExcludedComp : int
        To avoid that spectral leaks from the signal affect the noise-power estimate,
        the `nExcludedComp` components just above and the `nExcludedComp` components just
        below the signal will not be used for estimating noise power.
    FFTDict: dict with the following keys
       - mag : array of floats
           The array containing the FFT magnitude values.
       - freq : array of floats
           The array containing the FFT frequencies.
    otherExclude : array of ints
        The frequencies of other components to exclude from the computation of the noise power.
        This may be useful to exclude components corresponding to distortion products
        generated by the signal. The `nExcludedComp` components just above and the
        `nExcludedComp` components just below each component in `otherExclude` will
        also be excluded.

    Returns
    ----------
    noiseBands : list
        The spectral magnitude of the noise bands. A separate sub-list is returned for
        each component specified in `freqs`. 
    noiseBandsIdx : list
        The indexes of the frequency bins in `fftDict` corresponding to the noise bands.
        A separate sub-list is returned for each component specified in `freqs`. 
    idxProtect : list
        The indexes of the frequency bins in `fftDict` that were excluded from
        the noise power computation.
    
    Examples
    ----------
    >>> getNoiseSidebands(compIdx=[40, 44], nSideComp=30, nExcludedComp=2, FFTDict=ffts, otherExclude=[36, 42])
    """
    
    compIdx = array([], dtype=int)
    for i in range(len(componentsFreq)):
        compIdx = numpy.append(compIdx, where(abs(fftDict['freq'] - componentsFreq[i]) == min(abs(fftDict['freq'] - componentsFreq[i])))[0][0])

        
    idxProtect = []; idxProtect.extend(compIdx);
    
    if otherExclude != None:
        otherExcludeIdx = array([], dtype=int)
        for i in range(len(otherExclude)):
            otherExcludeIdx = numpy.append(otherExcludeIdx, where(abs(fftDict['freq'] - otherExclude[i]) == min(abs(fftDict['freq'] - otherExclude[i])))[0][0])
        idxProtect.extend(otherExcludeIdx)
        
    for i in range(nExcludedComp):
        idxProtect.extend(numpy.array(compIdx) + (i+1))
        idxProtect.extend(numpy.array(compIdx) - (i+1))
        for j in range(len(otherExclude)):
            idxProtect.extend([otherExcludeIdx[j] + (i+1)])
            idxProtect.extend([otherExcludeIdx[j] - (i+1)])
    noiseBands = []; noiseBandsIdx = []

    for i in range(len(compIdx)):
        loSide = []; loSideIdx = []
        hiSide = []; hiSideIdx = []
        counter = 1
        while len(hiSide) < nCompSide:
            currIdx = compIdx[i] + nExcludedComp + counter
            if currIdx not in idxProtect:
                hiSide.append(fftDict['mag'][currIdx])
                hiSideIdx.append(currIdx)
            counter = counter + 1
            
        counter = 1
        while len(loSide) < nCompSide:
            currIdx = compIdx[i] - nExcludedComp - counter
            if currIdx not in idxProtect:
                loSide.append(fftDict['mag'][currIdx])
                loSideIdx.append(currIdx)
            counter = counter + 1
        noiseBands.append(loSide+hiSide)
        noiseBandsIdx.append(loSideIdx+hiSideIdx)
    return noiseBands, noiseBandsIdx, idxProtect


def mergeTriggersCnt(trigArray, trigList, newTrig):
    """
    Take one or more triggers in trigList, and substitute them with newTrig

    Parameters
    ----------
    trigArray : array
        The trigger channel.
    trigList : array 
        The list of triggers that should be substituted with `newTrig`
    newTrig : 
        The new trigger value.

    Examples
    ----------
    >>> mergeTriggersCnt(trigArray, [1,2], 100)
    
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

    Examples
    ----------
    >>> removeEpochs(rec, toRemove)
    
    """
    eventList = list(rec.keys())
    #print(toRemove)
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
    """
    Remove from the eventTable triggers that were not actually sent.
    """
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
    rec : array of floats
        The nChannelsXnSamples array with the EEG data.
    refChannel: int
        The reference channel (indexing starts from zero).
    channels : list of ints
        List of channels to be rereferenced (indexing starts from zero).
  
    Examples
    --------
    >>> rerefCnt(rec=dats, refChannel=4, channels=[1, 2, 3])
    """

    if channels == None:
        nChannels = rec.shape[0]
        channels = list(range(nChannels))

    rec[channels,:] = rec[channels,:] - rec[refChannel,:]

    return

def rerefSegmented(rec, refChannel, channels=None):
    """
    Rereference channels in a segmented recording.

    Parameters
    ----------
    rec : dict of 3D arrays
        The segmented recording
    refChannel: int
        The reference channel (indexing starts from zero).
    channels : list of ints
        List of channels to be rereferenced (indexing starts from zero).
  
    Examples
    --------
    >>> rerefSegmented(rec=segs, refChannel=4, channels=[0,1])
    """
    
    keyList = list(rec.keys())
    if channels == None:
        nChannels = rec[keyList[0]].shape[0]
        channels = list(range(nChannels))
    for k in keyList:
        rec[k][channels,:,:] = rec[k][channels,:,:] - rec[k][refChannel,:,:]

    return 



def segmentCnt(rec, eventTable, epochStart, epochEnd, sampRate, eventList=None):
    """
    Segment a continuous EEG recording into discrete event-related epochs.
    
    Parameters
    ----------
    rec: array of floats
        The nChannelsXnSamples array with the EEG data.
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
    Compute the exponent of the closest power of two that is either equal
    to `x` of bigger than `x`.
    
    Parameters
    ----------
    x : numeric
    
    Returns
    ----------
    y : numeric
    
    Examples
    ----------
    >>> nextPowTwo(7)
    >>> nextPowTwo(8)
    
    """
    out = int(ceil(log2(x)))
    return out

## def getFFT(sig, sampRate, window, powerOfTwo):
##     """
##     Compute the fast fourier transform of a 1-dimensional array.
    
##     Parameters
##     ----------

##     Returns
##     ----------

##     Examples
##     ----------
##     """
##     n = len(sig)
##     if powerOfTwo == True:
##         nfft = 2**nextPowTwo(n)
##     else:
##         nfft = n
##     if window != 'none':
##         if window == 'hamming':
##              w = hamming(n)
##         elif window == 'hanning':
##              w = hanning(n)
##         elif window == 'blackman':
##              w = blackman(n)
##         elif window == 'bartlett':
##              w = bartlett(n)
##         sig = sig*w
        
##     p = fft(sig, nfft) # take the fourier transform 
##     nUniquePts = ceil((nfft+1)/2.0)
##     p = p[0:nUniquePts]
##     p = abs(p)
##     p = p / sampRate  # scale by the number of points so that
##     # the magnitude does not depend on the length 
##     # of the signal or on its sampling frequency  
##     #p = p**2  # square it to get the power 

##     # multiply by two (see technical document for details)
##     # odd nfft excludes Nyquist point
##     if nfft % 2 > 0: # we've got odd number of points fft
##          p[1:len(p)] = p[1:len(p)] * 2
##     else:
##          p[1:len(p) -1] = p[1:len(p) - 1] * 2 # we've got even number of points fft

##     freqArray = arange(0, nUniquePts, 1.0) * (sampRate / nfft);
##     x = {'freqArray': freqArray, 'mag':p}
##     return x

def getSpectrogram(sig, sampRate, winLength, overlap, winType, powerOfTwo):
    """
    Compute the spectrogram of a 1-dimensional array.
    
    Parameters
    ----------
    sig : array of floats
        The signal of which the spectrum should be computed.
    sampRate : int
        The sampling rate of the signal.
    winLength : float
        The length of the window over which to take the FFTs.
    overlap : float
        The percent of overlap between successive windows (useful for smoothing the spectrogram).
    winType : str {'hamming', 'hanning', blackman', 'bartlett', 'none'}
        The type of window to apply to the signal before computing its FFT.
        Choose 'none' if you don't want to apply any window.
    powerOfTwo : bool
        If `True` `sig` will be padded with zeros (if necessary) so that its length is a power of two.
        
    Returns
    ----------
    spectrogram : dict with the following keys
                  - freq : array of floats
                      The frequency axis.
                  - time : array of floats
                      The time axis.
                  - mag : the power spectrum.

    Examples
    ----------
    >>> sig = np.random.random(512)
    >>> getSpectogram(sig, 256, 'hamming')
    """
    """
    Compute the spectrogram of a 1-dimensional array.
    
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
        x = getSpectrum(sig[ind[i]:ind[i]+winLengthPnt], sampRate, winType, powerOfTwo)
        freq_array = x['freq']; p = x['mag']
        power_matrix[:,i] = p

    timeInd = arange(0, len(sig), step)
    time_array = 1./sampRate * (timeInd)
    x = {'freq': freq_array, 'time': time_array, 'mag': power_matrix}
    return x

def getSpectrum(sig, sampRate, window, powerOfTwo):
    """
    Compute the power spectrum of a 1-dimensional array.
    
    Parameters
    ----------
    sig : array of floats
        The signal of which the spectrum should be computed.
    sampRate : int
        The sampling rate of the signal.
    window : str {'hamming', 'hanning', blackman', 'bartlett', 'none'}
        The type of window to apply to the signal before computing its FFT.
        Choose 'none' if you don't want to apply any window.
    powerOfTwo : bool
        If `True` `sig` will be padded with zeros (if necessary) so that its length is a power of two.
        
    Returns
    ----------
    spectrum : dict with the following keys
                  - freq : array of floats
                      The FFT frequencies.
                  - mag : the power spectrum.

    Examples
    ----------
    >>> sig = np.random.random(512)
    >>> getSpectrum(sig, 256, 'hamming')
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


def tukeywin(window_length, alpha=0.5):
    ## taken from http://leohart.wordpress.com/2006/01/29/hello-world/
    ## The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    ## that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    ## at \alpha = 0 it becomes a Hann window.
 
    ## We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    ## output
 
    ## Reference
    ## ---------
 
    ## http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
 
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w
