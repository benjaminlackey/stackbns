######################################################
# Tools for manipulating results of MCMC runs.
######################################################
# This temporary file is the same as mcmcutilities.py but without any plotting functions.

import numpy as np
import scipy.ndimage.filters as filters # for Gaussian filter
import sys
# you'll have to fix this so you don't rely on the EOS modules:
import eos4parameterpolytrope as eos4p
import binaryutilities as binary

########## useful constants #############

# negative number with large magnitude used to represent the log of zero
logZero = -sys.float_info.max * sys.float_info.epsilon

sigma1 = 0.68268949 # ~317 points out of 1000 are outside
sigma2 = 0.95449974 # ~46 points out of 1000 are outside
sigma3 = 0.99730024 # ~3 points out of 1000 are outside


############ utilities for emcee MCMC program ################

def flatten_emcee_chain(emceeChain, nBurn=0, dSteps=1):
    prunedChain = emceeChain[:, nBurn::dSteps, :]
    nWalkers, nStepsLeft, nDim = prunedChain.shape
    
    return np.reshape(prunedChain, (nWalkers*nStepsLeft, nDim))


###################### 1-dimensional histogram functions ######################


def CenteredHistogram1D(samplesx, bins, histrange=None, density=False, smooth=False):
    """Takes a list of MCMC samples then bins them.
        Returns the 1d-histogram array, and
        1d array of the x-values centered on each bin.
        """
    
    # set histogram range if not given explicitly
    if histrange is None:
        histrange = [np.min(samplesx), np.max(samplesx)]
    
    hist, xedges = np.histogram(samplesx, bins=bins, range=histrange, density=density)
    
    # Gaussian smoothing
    if smooth:
        # Smooth the hist2d object with a Gaussian filter. Return values
        # at the same points and replace hist2d with those values.
        hist = filters.gaussian_filter(hist, sigma=0.75)

    # Delete the last point then shift the other points up half a bin to give the midpoint of each bin
    xcenters = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    return hist, xcenters


def MultiplyChains1D(sampleslist, bins, histrange=None, individualsmooth=False, productsmooth=False):
    """Multiply together the marginalized posteriors (PDFs) produced by multiple MCMC runs.
        
        Takes a list of 1-parameter from MCMC runs, then histograms each run, then multiplies the histograms together.
        The final result is normalized.
        
        If you have non-flat priors, you will have to later divide by prior^(N_chains-1)
        to get a final posterior.
        """
    
    # Set histogram range if not given explicitly.
    # min = smallest value in any of the chains.
    # max = largest value in any of the chains.
    if histrange is None:
        # np.min and np.max search each element of each level
        histrange = [np.min(sampleslist), np.max(sampleslist)]

    #print(histrange)

    # Histogram each chain.
    # Then set all 0 counts to 1.
    # Then smooth histogram for each chain if individualsmooth=True.
    hist1dlist = []
    for n in range(len(sampleslist)):
        samples = np.array(sampleslist)[n, :]
        # You want the count instead of the density.
        # xcenters will be the same for each chain (only have to store 1 list).
        hist, centers = CenteredHistogram1D(samples, bins, histrange=histrange, density=False, smooth=False)
        
        # Set all instances of 0 to 1 in bins, because you want to
        # multiply histograms together later, and the result
        # will be 0 at any point where any of the histograms are 0.
        for i, value in np.ndenumerate(hist):
            if hist[i] == 0:
                hist[i] = 1
        
        # Gaussian smoothing of individual histograms
        if individualsmooth:
            hist = filters.gaussian_filter(hist, sigma=0.75)
        
        #fig = plt.figure(figsize=(4, 3))
        #axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        #axes1.plot(centers, hist, color='k', lw=1.5)
                
        hist1dlist.append(hist)

    # Multiply all the histograms together
    hist1dproduct = hist1dlist[0]
    for n in range(1, len(hist1dlist)):
        # * multiplies N-d arrays element-wise
        hist1dproduct = hist1dproduct * hist1dlist[n]

    # Gaussian smoothing of product
    if productsmooth:
        hist1dproduct = filters.gaussian_filter(hist1dproduct, sigma=0.75)

    # Normalize the result
    norm = bins / ((histrange[1]-histrange[0])*np.sum(hist1dproduct))
    hist1dproduct = hist1dproduct * norm

    return hist1dproduct, centers


# This is a stupid function. You should just have a function called Rescale that sets the max value to newmax and multiplies all other points by newmax/max
def MatchHistogramScale1D(histlist, newmax=1.0):
    """Rescale each histogram in the list,
        so they all have the same maximum value.
        """
    
    histrescaledlist = np.zeros(np.array(histlist).shape)
    
    for n in range(len(histlist)):
        oldmax = np.max(histlist[n])
        histrescaledlist[n] = histlist[n]*newmax/oldmax
    
    return histrescaledlist

##################################################################################################
#!!! The index nIndex goes out of bounds when all the elements in hist are 0.0 !!!               #
# This can happen when the histrange does not include the points where the histogram is nonzero. #
##################################################################################################
def GetConfidenceInterval1D(hist, xcenters, frac):
    """Takes a 1d-histogram array and the frac (between 0 and 1).
        Returns the value of the contour that contains frac fraction of the points.
        The histogram can be the number of points, or be the normalized PDF,
        or have any other overall constant factor.
        """
    # then sort the bins from smallest to greatest
    sortpost = np.sort(hist)
    
    # sum all the bins
    dTotal = np.sum(sortpost)
    # start at last element then add the bins until you get to frac fraction of the total
    nIndex = sortpost.size # shouldn't this start at sortpost.size-1?
    dSum = 0
    while (dSum < dTotal * frac):
        nIndex -= 1
        dSum += sortpost[nIndex]
    level = sortpost[nIndex]
    
    # now find the interval that contains this level
    # currently assumes single mode
    # could look for all the points where (hist[i]-level)*(hist[i+1]-level) is negative
    # or all the points where hist[i+1]-level changes sign from hist[i]-level
    i=0
    while hist[i] < level:
        i += 1
    xmin = xcenters[i]
    i = len(hist)-1
    while hist[i] < level:
        i -= 1
    xmax = xcenters[i]

    return xmin, xmax, level


################################################################################
###################### 2-dimensional histogram functions #######################
################################################################################


def CenteredHistogram2D(samplesx, samplesy, bins, histrange=None, smooth=False):
    """Takes a list of MCMC samples then bins them.
        Returns the 2d-histogram array of z-values, and 2 1d arrays of the
        x-values and y-values of the center of each bin.
        """
    
    # set histogram range if not given explicitly
    if histrange is None:
        histrange = [[np.min(samplesx), np.max(samplesx)], [np.min(samplesy), np.max(samplesy)]]
    
    # edges give the coordinate at the edges of the bins
    # for N bins, there are N+1 edges
    hist2d, xedges, yedges = np.histogram2d(samplesx, samplesy, bins=bins, range=histrange)

    # Gaussian smoothing
    if smooth:
        hist2d = filters.gaussian_filter(hist2d, sigma=0.75)

    # np.delete(xedges, -1) generates a copy of xedges that dosen't have the point with index -1
    # it doesn't actually delete the point from xedges
    # delete the last point then shift the other points up half a bin to give the midpoint of each bin
    xcenters = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    ycenters = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])
    
    return hist2d, xcenters, ycenters


def MultiplyChains2D(sampleslistx, sampleslisty, bins, histrange=None, individualsmooth=False, productsmooth=False):
    """Multiply together the marginalized posteriors (PDFs) produced by multiple MCMC runs.
        
        Takes two lists of parameters from MCMC runs, then histograms each run, then multiplies the histograms together.
        The final result is normalized.
        
        If you have non-flat priors, you will have to later divide by prior^(N_chains-1)
        to get a final posterior.
        """
    
    # Set histogram range if not given explicitly.
    # min = smallest value in any of the chains.
    # max = largest value in any of the chains.
    if histrange is None:
        # np.min and np.max search each element of each level
        histrange = [[np.min(sampleslistx), np.max(sampleslistx)], [np.min(sampleslisty), np.max(sampleslisty)]]
    
    #print(histrange)
    
    # Histogram each chain.
    # Then set all 0 counts to 1.
    # Then smooth histogram for each chain if individualsmooth=True.
    hist2dlist = []
    for n in range(len(sampleslistx)):
        samplesx = np.array(sampleslistx)[n, :]
        samplesy = np.array(sampleslisty)[n, :]
        # You want the count instead of the density.
        # xcenters and ycenters will be the same for each chain.
        hist2d, xcenters, ycenters = CenteredHistogram2D(samplesx, samplesy, bins, histrange=histrange, smooth=False)
        
        # Set all instances of 0 to 1 in bins, because you want to
        # multiply histograms together later, and the result
        # will be 0 at any point where any of the histograms are 0.
        for (i, j), value in np.ndenumerate(hist2d):
            if hist2d[i, j] == 0:
                hist2d[i, j] = 1
        
        # Gaussian smoothing of individual histograms
        if individualsmooth:
            hist2d = filters.gaussian_filter(hist2d, sigma=0.75)
        
        hist2dlist.append(hist2d)
    
    # Multiply all the histograms together
    hist2dproduct = hist2dlist[0]
    for n in range(1, len(hist2dlist)):
        # * multiplies N-d arrays element-wise
        hist2dproduct = hist2dproduct * hist2dlist[n]
    
    # Gaussian smoothing of product
    if productsmooth:
        hist2dproduct = filters.gaussian_filter(hist2dproduct, sigma=0.75)

    # Normalize the result
    norm = bins*bins / ((histrange[0][1]-histrange[0][0])*(histrange[1][1]-histrange[1][0])*np.sum(hist2dproduct))
    hist2dproduct = hist2dproduct * norm
    
    return hist2dproduct, xcenters, ycenters


def GetConfidenceContourLevel(hist2d, frac):
    """Takes a 2d-histogram array and the frac (between 0 and 1).
        Returns the value of the contour that contains frac fraction of the points.
        The histogram can be the number of points, or be the normalized PDF,
        or have any other overall constant factor.
        """
    # reshape the histogram to a 1d array of length hist2d.size
    # then sort the bins from smallest to greatest
    post = hist2d.reshape(hist2d.size)
    sortpost = np.sort(post)
    
    # sum all the bins
    dTotal = np.sum(sortpost)
    # start at last element then add the bins until you get to sigma1 fraction of the total
    nIndex = sortpost.size
    dSum = 0
    while (dSum < dTotal * frac):
        nIndex -= 1
        dSum += sortpost[nIndex]
    level = sortpost[nIndex]
    
    return level


def Confidence2DFromChain(samplesx, samplesy, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False):
    """Takes x and y samples from an MCMC chain, list of confidence values, #bins in each direction.
        Creates 2d-histogram, smooths it, and finds the levels corresponding to the confidence values.
        Returns hist2d, xcenters, ycenters, level.
        """

    #hist2d, xcenters, ycenters = CenteredHistogram2D(samplesx, samplesy, bins, histrange)
    hist2d, xcenters, ycenters = CenteredHistogram2D(samplesx, samplesy, bins, histrange=histrange, smooth=smooth)
    # Smooth the hist2d object with a Gaussian filter. Return values
    # at the same points and replace hist2d with those values.
    #hist2d = filters.gaussian_filter(hist2d, sigma=0.75)
    
    # Can this also be done with the map function? It has 2 arguments. How is that done?
    level = [0.0]*len(conf)
    for i in range(len(conf)):
        level[i] = GetConfidenceContourLevel(hist2d, conf[i])
    
    return hist2d, xcenters, ycenters, level


################################################################################
###################### N-dimensional histogram functions #######################
################################################################################


def CenteredHistogramND(samples, bins, histrange=None, smooth=False):
    """Takes a list of MCMC samples then bins them.
        Returns the 2d-histogram array of z-values, and 2 1d arrays of the
        x-values and y-values of the center of each bin.
        """
    
    # edges give the coordinate at the edges of the bins
    # edges has n rows (1 for each dimension) with bins+1 elements in each row
    # range defaults to [[minx maxx], [miny maxy], [minz maxz],...] if None
    histnd, edges = np.histogramdd(samples, bins=bins, range=histrange)
    
    # Gaussian smoothing
    if smooth:
        histnd = filters.gaussian_filter(histnd, sigma=0.75)
    
    centers = []
    for n in range(len(edges)):
        centersn = np.delete(edges[n], -1) + 0.5*(edges[n][1] - edges[n][0])
        centers.append(centersn)
    
    return histnd, np.array(centers)


