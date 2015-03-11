######################################################
# Tools for manipulating results of MCMC runs.
######################################################


import numpy as np
import scipy.ndimage.filters as filters # for Gaussian filter
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator, MaxNLocator
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

def FlattenEmceeChain(emceeChain, nBurn=0, dSteps=1):
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


#def ConfidencePlot1DFromChain(axes, samplesx, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False):
#    """Draw confidence plot. Includes scatter plot of MCMC points, histogram of MCMC points, and confidence contours.
#        """
#    
#    assert (len(conf)<=3), "Number of contours should be <=3"
#    
#    # set histogram range if not given explicitly
#    if histrange is None:
#        histrange = [np.min(samplesx), np.max(samplesx)]
#    
#    hist, xcenters = CenteredHistogram1D(samplesx, bins, histrange=histrange, density=False, smooth=smooth)
#    histrescale = MatchHistogramScale1D([hist])[0]
#    
#    # plot rescaled histogram
#    axes.plot(xcenters, histrescale, color='k', lw=1.5)
#    
#    # find and plot each confidence interval
#    colorlist = ['red', 'darkgreen', 'blue']
#    for n in range(len(conf)):
#        # get confidence intervals
#        xmin, xmax, level = GetConfidenceInterval1D(histrescale, xcenters, conf[n])
#        
#        # plot confidence boundaries
#        axes.plot([xmin, xmin], [0, level], color=colorlist[n], ls='-', lw=1.5)
#        axes.plot([xmax, xmax], [0, level], color=colorlist[n], ls='-', lw=1.5)
#        
#        # fill confidence region
#        axes.fill_between(xcenters, 0.0, histrescale, where=histrescale>=level, facecolor='darkgrey', alpha=0.5)
#        #axes.fill_between(xcenters, 0.0, histrescale, where=xmin<=xcenters, facecolor='grey', alpha=0.5)



#######################################################
# Temporary function until you clean things up        #
#######################################################
def ConfidencePlot1DFromChain(axes, samplesx, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, \
                              contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """Draw confidence plot. Includes scatter plot of MCMC points, histogram of MCMC points, and confidence contours.
        """
    
    assert (len(conf)<=3), "Number of contours should be <=3"
    
    # set histogram range if not given explicitly
    if histrange is None:
        histrange = [np.min(samplesx), np.max(samplesx)]
    
    hist, xcenters = CenteredHistogram1D(samplesx, bins, histrange=histrange, density=False, smooth=smooth)
    histrescale = MatchHistogramScale1D([hist])[0]
    
    ## plot rescaled histogram
    #axes.plot(xcenters, histrescale, color='k', lw=1.5)
    
    # find and plot each confidence interval
    for n in range(len(conf)):
        ## get confidence intervals
        #xmin, xmax, level = GetConfidenceInterval1D(histrescale, xcenters, conf[n])
        
        ## plot confidence boundaries
        #axes.plot([xmin, xmin], [0, level], color=colorlist[n], ls='-', lw=1.5)
        #axes.plot([xmax, xmax], [0, level], color=colorlist[n], ls='-', lw=1.5)
        
        ## fill confidence region
        #axes.fill_between(xcenters, 0.0, histrescale, where=histrescale>=level, facecolor='darkgrey', alpha=0.5)
        ##axes.fill_between(xcenters, 0.0, histrescale, where=xmin<=xcenters, facecolor='grey', alpha=0.5)
        axes.plot(xcenters, histrescale, c=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])


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


def ConfidencePlot2DFromChain(axes, samplesx, samplesy, bins, conf=[sigma1, sigma2, sigma3], maxpoints=0, histrange=None, showhist=True, histsmooth=False, confsmooth=True, colorbar=False, contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Draw confidence plot. Includes scatter plot of MCMC points, histogram of MCMC points, and confidence contours.
        conf=[sigma1, sigma2, sigma3]: confidence intervals to plot
        maxpoints=0: maximum number of points to plot in scatter plot. Will never be more than number of points available. Can specify 'all'
        histrange=None: max and min values of bins for making histogram and contour plots [[xmin, xmax], [ymin, ymax]]
        showhist=True: show the 2-d histogram of the bins
        histsmooth=False: Gaussian smoothing for the histogram plot
        confsmooth=False: Gaussian smoothing for making the confidence contours
        colorbar=False: scale for number of points in each bin (out of all the samples not just the maxpoints value)
        """
    
    assert (len(conf)<=3), "Number of contours should be <=3"
    
    # set histogram range if not given explicitly
    if histrange is None:
        histrange = [[np.min(samplesx), np.max(samplesx)], [np.min(samplesy), np.max(samplesy)]]
    
    ############# Make the scatter plot if requested by maxpoints being>0 #############
    
    if maxpoints > 0:
        if maxpoints == 'all':
            # plot all the points
            resamplex, resampley = samplesx, samplesy
        else:
            # resample to provide maxpoints points for the scatter plot
            di = max(1, len(samplesx)/maxpoints) # Prevent di from being 0
            imaxp1 = min(len(samplesx), maxpoints*di) # maximum value of len(samplesx)
            resamplex = samplesx[0:imaxp1:di]
            resampley = samplesy[0:imaxp1:di]
        
        axes.scatter(resamplex, resampley, s=1, facecolor='0.0', lw = 0)
    
    ############# Make the histogram plot if requested #############
    
    if showhist:
        # Calculate 2D histogram to plot
        hist2d, xcenters, ycenters = CenteredHistogram2D(samplesx, samplesy, bins, histrange=histrange, smooth=histsmooth)
        
        # Image of histogram
        extent = [histrange[0][0], histrange[0][-1], histrange[1][0], histrange[1][-1]]
        im = axes.imshow(np.flipud(hist2d.T), cmap='BuGn', interpolation='none', extent=extent, aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))
        if colorbar:
            plt.colorbar(mappable=im, ax=axes)
    
    ############# Calculate and make the confidence contours #############
    
    # Calculate 2D histogram and confidence levels for making contour plot
    hist2dconf, xcenters, ycenters, levels = Confidence2DFromChain(samplesx, samplesy, bins, conf=conf, histrange=histrange, smooth=confsmooth)
    
    for n in range(len(levels)):
        cs = axes.contour(xcenters, ycenters, hist2dconf.T, [levels[n]], colors=contourcolors[n], linestyles=contourlinestyles[n], linewidths=contourlinewidths[n], zorder=2)


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


################################################################################
#    Triangle plot for showing marginalized distributinos and correlations.    #
################################################################################


# make triangle plot of marginalized posterior distribution
def TrianglePlot(fig, chain, bins, histrange=None, conf=[sigma1, sigma2, sigma3], maxpoints=1000, showhist=True, truevalue=None, labels=None, \
                 contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    
    """
        Make Triangle plot
        """
    
    # get number of parameters
    ndim = chain.shape[1]
    
    for i in range(ndim): # row number
        for j in range(i+1): # column number (row i goes up to column i for triangular figure)
            
            # get subplot index number which is indexed sequentially from left to right then top to bottom starting at 1 (not 0)
            spn = i*ndim + j + 1
            axn = fig.add_subplot(ndim, ndim, spn)
            
            xmajorLocator = MaxNLocator(nbins=4,prune='both')
            #ymajorLocator = MaxNLocator(nbins=4,prune='both')
            
            # the column number j defines the x axis parameter index
            xind = j
            yind = i
            # the row number i defines the y axis parameter index
            if j == i:
                ################# Make a 1D plot #################
                # Get range for 1-d histogram if given
                if histrange:
                    histrange1d = histrange[xind]
                else:
                    histrange1d = None
                # Make histogram
                ConfidencePlot1DFromChain(axn, chain[:, xind], bins, conf=conf, histrange=histrange1d, smooth=True, \
                                          contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
                # get rid of y-axis numbers
                axn.yaxis.set_major_formatter(NullFormatter())
                # plot true value if given
                if truevalue:
                    axn.plot([truevalue[xind], truevalue[yind]], [0, 1.0], color='k', ls='-', lw=1.5)
            else:
                ################# Make a 2D plot #################
                if histrange:
                    histrange2d = np.array(histrange)[[xind, yind]].tolist() # !!!find better way to index this!!!
                else:
                    histrange2d = None
                ConfidencePlot2DFromChain(axn, chain[:, xind], chain[:, yind], bins, \
                                          conf=conf, maxpoints=maxpoints, histrange=histrange2d, showhist=showhist, \
                                          histsmooth=False, confsmooth=True, colorbar=False, \
                                          contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
                # plot true value if given
                if truevalue:
                    axn.scatter([truevalue[xind]], [truevalue[yind]], facecolor="k", marker="x", s=200, lw = 3)
            
            # get rid of x axes numbers
            if i != ndim-1:
                axn.xaxis.set_major_formatter(NullFormatter())
            # get rid of y axes numbers
            if j != 0:
                axn.yaxis.set_major_formatter(NullFormatter())
            
            # label the x-axis only for the bottom figures
            if i == ndim-1:
                axn.set_xticklabels(axn.get_xticks(), fontsize=12)
                #axn.xaxis.set_major_formatter(xmajorLocator)
                if labels:
                    axn.set_xlabel(labels[xind], fontsize=18)
            
            # label y-axis for left column except for top (the 1-parameter histogram)
            if j == 0 and i != 0:
                axn.set_yticklabels(axn.get_yticks(), fontsize=12)
                if labels:
                    axn.set_ylabel(labels[yind], fontsize=18)
    
    # make plots closer together
    fig.subplots_adjust(hspace=0.12)
    fig.subplots_adjust(wspace=0.12)


## make triangle plot of marginalized posterior distribution
#def TrianglePlot(chain, bins, histrange=None, conf=[sigma1, sigma2, sigma3], maxpoints=1000, showhist=True, truevalue=None, labels=None, figsize=(11, 11), title=None):
#    
#    """
#    Make Triangle plot
#    """
#    
#    # rcParams settings
#    plt.rcParams['ytick.labelsize'] = 10.0
#    plt.rcParams['xtick.labelsize'] = 10.0
#    plt.rcParams['text.usetex'] = True
#    plt.rcParams['figure.figsize'] = figsize
#    
#    # get number of parameters
#    ndim = chain.shape[1]
#    parameters = np.linspace(0,ndim-1,ndim)
#    
#    f, axarr = plt.subplots(nrows=len(parameters), ncols=len(parameters),figsize=figsize)
#    
#    for i in range(len(parameters)):
#        # for j in len(parameters[np.where(i <= parameters)]:
#        for j in range(len(parameters)):
#            ii = i
#            jj = len(parameters) - j - 1
#            
#            xmajorLocator = MaxNLocator(nbins=4,prune='both')
#            ymajorLocator = MaxNLocator(nbins=4,prune='both')
#            
#            if j <= len(parameters)-i-1:
#                axarr[jj][ii].xaxis.set_minor_locator(NullLocator())
#                axarr[jj][ii].yaxis.set_minor_locator(NullLocator())
#                axarr[jj][ii].xaxis.set_major_locator(NullLocator())
#                axarr[jj][ii].yaxis.set_major_locator(NullLocator())
#                
#                axarr[jj][ii].xaxis.set_minor_formatter(NullFormatter())
#                axarr[jj][ii].yaxis.set_minor_formatter(NullFormatter())
#                axarr[jj][ii].xaxis.set_major_formatter(NullFormatter())
#                axarr[jj][ii].yaxis.set_major_formatter(NullFormatter())
#                xmajorFormatter = FormatStrFormatter('%g')
#                ymajorFormatter = FormatStrFormatter('%g')
#                
#                if ii == jj:
#                    ################# Make a 1D plot #################
#                    if histrange:
#                        histrange1d = histrange[ii]
#                    else:
#                        histrange1d = None
#                    ConfidencePlot1DFromChain(axarr[ii][jj], chain[:,parameters[ii]], bins, conf=conf, histrange=histrange1d, smooth=True)
#                    # plot true value if given
#                    if truevalue:
#                        axarr[jj][ii].plot([truevalue[ii], truevalue[ii]], [0, 1.0], color='k', ls=':', lw=1.5)
#                
#                else:
#                    ################# Make a 2D plot #################
#                    if histrange:
#                        histrange2d = np.array(histrange)[[ii, jj]].tolist() # !!!find better way to index this!!!
#                    else:
#                        histrange2d = None
#                    ConfidencePlot2DFromChain(axarr[jj][ii], chain[:, parameters[ii]], chain[:, parameters[jj]], bins, \
#                                              conf=conf, maxpoints=maxpoints, histrange=histrange2d, showhist=showhist, \
#                                              histsmooth=False, confsmooth=True, colorbar=False)
#                    # plot true value if given
#                    if truevalue:
#                        axarr[jj][ii].scatter([truevalue[ii]], [truevalue[jj]], facecolor="k", marker="x", s=200, lw = 3)
#                
#                axarr[jj][ii].xaxis.set_major_locator(xmajorLocator)
#                axarr[jj][ii].yaxis.set_major_locator(ymajorLocator)
#            else:
#                axarr[jj][ii].set_visible(False)
#            
#            if jj == len(parameters)-1:
#                axarr[jj][ii].xaxis.set_major_formatter(xmajorFormatter)
#                if labels:
#                    axarr[jj][ii].set_xlabel(labels[ii])
#            
#            if ii == 0:
#                if jj == 0:
#                    axarr[jj][ii].yaxis.set_major_locator(NullLocator())
#                #axarr[jj][ii].set_ylabel('Post.')
#                else:
#                    axarr[jj][ii].yaxis.set_major_formatter(ymajorFormatter)
#                    if labels:
#                        axarr[jj][ii].set_ylabel(labels[jj])
#    
#    # overall plot title
#    if title:
#        f.suptitle(title, fontsize=14, y=0.90)
#    
#    # make plots closer together
#    f.subplots_adjust(hspace=0.1)
#    f.subplots_adjust(wspace=0.1)

