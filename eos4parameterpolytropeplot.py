import numpy as np
import mcmcutilities as mu
from mcmcutilities import sigma1, sigma2, sigma3
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator, MaxNLocator
from matplotlib.colors import Normalize, LogNorm
import eos4parameterpolytrope as eos4p
import binaryutilities as binary



################################################################################
########### Make NS structure plots for 4-parameter EOS ########################
################################################################################


def MaxMassConfidencePlot(axes, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False):
    """
        Calculate confidence interval for maximum mass and generate histogram.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    maxmass = np.zeros(len(eoschain))
    for i in range(len(maxmass)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        maxmass[i] = eos4p.MMaxOfP123(lp, g1, g2, g3)
    
    mu.ConfidencePlot1DFromChain(axes, maxmass, bins, conf=conf, histrange=histrange, smooth=smooth)


################ Radius functions ##################


# It would be better to store R, k2, etc. during the MCMC run
# (I think emcee allows storing of metadata) instead of recalculating it here.
# But, it's not calculated for a uniformly spaced list of masses during the MCMC run,
# so maybe it doesn't matter.
def RadiusConfidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
    """
        Calculate radius confidence interval for given mass.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    radius = np.zeros(len(eoschain))
    for i in range(len(radius)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
            radius[i] = eos4p.ROfP123M(lp, g1, g2, g3, mass)
        else:
            radius[i] = 0.0
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(radius, bins, histrange=histrange, density=False, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlist = np.zeros((len(conf), 2))
    for n in range(len(conf)):
        rmin, rmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
        minmaxlist[n, [0, 1]] = [rmin, rmax]
    
    ## if rmin corresponds to unstable r=0 NS, find the true rmin
    #rtruemin = 0.0
    #if rmin < 2.0: # the rmin value is at the center of the bin, so rmin won't actually be at 0.0
    #    rtruemin = # the next r-value greater than rmin
    
    # plot histogram and confidence intervals if requested
    if showhist:
        fig = plt.figure(figsize=(4, 3))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        mu.ConfidencePlot1DFromChain(axes, radius, bins, conf=conf, histrange=histrange, smooth=smooth)
        axes.set_xlim([-1.0, 25.0])
    
    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds


def RadiusConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False, showhistimage=True, \
                            contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Plot R confidence bounds as a function of M.
        """
    
    marray = np.linspace(massmin, massmax, nmass)
    minmaxlist = []
    histlist = []
    for mass in marray:
        minmax, hist, centers = RadiusConfidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
        minmaxlist.append(minmax)
        histlist.append(hist)
    # centers is the same for all masses so don't make an list of them
    
    # plot each confidence interval
    for n in range(len(conf)):
        # truncate the list to m < mMax
        mrlower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 2.0])
        mrupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 2.0])
        axes.plot(mrlower[:, 0], mrlower[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
        axes.plot(mrupper[:, 0], mrupper[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
    
    # Image of histogram
    if showhistimage:
        extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
        vmax = 10*len(eoschain)/bins
        im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
                         aspect=axes.get_aspect(), norm=Normalize(vmin=0, vmax=vmax))
#im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
#                 aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))


############# lambda = 2/3G k_2 R^5 (g cm^2 s^2) functions ##############

def LambdaConfidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
    """
        Calculate lambda confidence interval for given mass.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    tidal = np.zeros(len(eoschain))
    for i in range(len(tidal)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
            tidal[i] = eos4p.LambdaOfP123M(lp, g1, g2, g3, mass)*(mass*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
        else:
            tidal[i] = 0.0
    #print(lp, g1, g2, g3, tidal[i])
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(tidal, bins, histrange=histrange, density=False, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlist = np.zeros((len(conf), 2))
    for n in range(len(conf)):
        tidalmin, tidalmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
        minmaxlist[n, [0, 1]] = [tidalmin, tidalmax]
    
    ## if rmin corresponds to unstable r=0 NS, find the true rmin
    #rtruemin = 0.0
    #if rmin < 2.0: # the rmin value is at the center of the bin, so rmin won't actually be at 0.0
    #    rtruemin = # the next r-value greater than rmin
    
    # plot histogram and confidence intervals if requested
    if showhist:
        fig = plt.figure(figsize=(4, 3))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        mu.ConfidencePlot1DFromChain(axes, tidal, bins, conf=conf, histrange=histrange, smooth=smooth)
        axes.set_xlim([-1.0, 20.0])
    
    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds


def LambdaConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False, showhistimage=True, \
                            contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Plot lambda confidence bounds as a function of M.
        """
    
    marray = np.linspace(massmin, massmax, nmass)
    minmaxlist = []
    histlist = []
    for mass in marray:
        minmax, hist, centers = LambdaConfidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
        minmaxlist.append(minmax)
        histlist.append(hist)
    # centers is the same for all masses so don't make an list of them
    
    # plot each confidence interval
    for n in range(len(conf)):
        # truncate the list to m < mMax
        mllower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 0.2])
        mlupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 0.2])
        axes.plot(mllower[:, 0], mllower[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
        axes.plot(mlupper[:, 0], mlupper[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
    
    # Image of histogram
    if showhistimage:
        extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
        vmax = 10*len(eoschain)/bins
        im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
                         aspect=axes.get_aspect(), norm=Normalize(vmin=0, vmax=vmax))
#im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
#                 aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))


############## Love number k_2 functions ###############

def K2Confidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
    """
        Calculate k2 confidence interval for given mass.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    tidal = np.zeros(len(eoschain))
    for i in range(len(tidal)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
            tidal[i] = eos4p.K2OfP123M(lp, g1, g2, g3, mass)
        else:
            tidal[i] = 0.0
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(tidal, bins, histrange=histrange, density=False, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlist = np.zeros((len(conf), 2))
    for n in range(len(conf)):
        tidalmin, tidalmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
        minmaxlist[n, [0, 1]] = [tidalmin, tidalmax]
    
    # plot histogram and confidence intervals if requested
    if showhist:
        fig = plt.figure(figsize=(4, 3))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        mu.ConfidencePlot1DFromChain(axes, tidal, bins, conf=conf, histrange=histrange, smooth=smooth)
        axes.set_xlim([-0.01, 0.15])
    
    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds


def K2ConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False, showhistimage=True, \
                        contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Plot k2 confidence bounds as a function of M.
        """
    
    marray = np.linspace(massmin, massmax, nmass)
    minmaxlist = []
    histlist = []
    for mass in marray:
        minmax, hist, centers = K2Confidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
        minmaxlist.append(minmax)
        histlist.append(hist)
    # centers is the same for all masses so don't make a list of them
    
    # plot each confidence interval
    for n in range(len(conf)):
        # truncate the list to m < mMax
        mllower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 0.01])
        mlupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 0.01])
        axes.plot(mllower[:, 0], mllower[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
        axes.plot(mlupper[:, 0], mlupper[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
    
    # Image of histogram
    if showhistimage:
        extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
        vmax = 10*len(eoschain)/bins
        im = axes.imshow(np.flipud(np.array(histlist).T), cmap='Greys', interpolation='none', extent=extent, \
                         aspect=axes.get_aspect(), norm=Normalize(vmin=1, vmax=vmax, clip=True))
#im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
#                 aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))


def StructurePlots(fig, lp, g1, g2, g3, massmin, massmax, nmass, eoschain, conf=[sigma1, sigma2, sigma3], showhistimage=True, \
                   contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Calculate structure quantities (Mmax, M--R, M--k_2, M--\lambda).
        lp, g1, g2, g3: true values.
        """
    
    ########### 1st plot #############
    
    ax1 = fig.add_subplot(221)
    
    histrange = [0.0, 4.0]
    dy = 0.05
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    print(bins)
    MaxMassConfidencePlot(ax1, eoschain, bins, conf=conf, histrange=histrange, smooth=True)
    truevalue = eos4p.MMaxOfP123(lp, g1, g2, g3)
    ax1.plot([truevalue, truevalue], [0, 1.1], color='k', ls=':', lw=1.5)
    ax1.set_xlim([0.0, 3.0])
    ax1.set_ylim([0.0, 1.1])
    ax1.set_xlabel(r'$M_{\rm max} (M_\odot)$', fontsize=18)
    ax1.set_xticklabels(ax1.get_xticks(), fontsize=12)
    ax1.yaxis.set_major_formatter(NullFormatter())
    
    ########### 2nd plot #############
    
    ax2 = fig.add_subplot(222)
    
    histrange=[0.0, 100.0]
    dy = 0.1
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    print(bins)
    # plot confidence bounds
    RadiusConfidenceOfMPlot(ax2, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=showhistimage, \
                            contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    
    # plot true EOS
    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
    ratmmax = eos4p.ROfP123M(lp, g1, g2, g3, eos4p.mmax) # R values past the max mass are set to R(M_max) in the interpolating table
    marray = np.linspace(massmin, massmax, nmass)
    mrtrue = np.array([[m, eos4p.ROfP123M(lp, g1, g2, g3, m)] for m in marray if m < mmax])
    mrtrue = np.append(mrtrue, [[mmax, ratmmax]], axis=0)
    ax2.plot(mrtrue[:, 0], mrtrue[:, 1], color='k', ls='-', lw=2.0)
    
    #bounds on masses
    ax2.plot([1.2, 1.2], [5, 20], color='k', ls=':', lw=1.5)
    ax2.plot([1.6, 1.6], [5, 20], color='k', ls=':', lw=1.5)
    
    ax2.set_xlim([0.0, 3.0])
    ax2.set_ylim([5.0, 20.0])
    ax2.set_xlabel(r'$M (M_\odot)$', fontsize=18)
    ax2.set_ylabel(r'$R$ (km)', fontsize=18)
    ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)
    
    ########### 3rd plot #############
    
    ax3 = fig.add_subplot(223)
    
    histrange=[0.0, 0.2]
    dy = 0.002
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    print(bins)
    # plot confidence bounds
    K2ConfidenceOfMPlot(ax3, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=showhistimage, \
                        contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    
    # plot true EOS
    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
    latmmax = eos4p.K2OfP123M(lp, g1, g2, g3, mmax)
    marray = np.linspace(massmin, massmax, nmass)
    mltrue = np.array([[m, eos4p.K2OfP123M(lp, g1, g2, g3, m)] for m in marray if m < mmax])
    mltrue = np.append(mltrue, [[mmax, latmmax]], axis=0)
    ax3.plot(mltrue[:, 0], mltrue[:, 1], color='k', ls='-', lw=2.0)
    
    #bounds on masses
    ax3.plot([1.2, 1.2], [0, 0.15], color='k', ls=':', lw=1.5)
    ax3.plot([1.6, 1.6], [0, 0.15], color='k', ls=':', lw=1.5)
    
    ax3.set_xlim([0.0, 3.0])
    ax3.set_ylim([0.0, 0.15])
    ax3.set_xlabel(r'$M (M_\odot)$', fontsize=18)
    ax3.set_ylabel(r'$k_2$', fontsize=18)
    ax3.set_xticklabels(ax3.get_xticks(), fontsize=12)
    ax3.set_yticklabels(ax3.get_yticks(), fontsize=12)
    
    ########### 4th plot #############
    
    ax4 = fig.add_subplot(224)
    
    histrange=[0.0, 20.0]
    dy = 0.1
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    print(bins)
    # plot confidence bound
    LambdaConfidenceOfMPlot(ax4, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=showhistimage, \
                            contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    
    # plot true EOS
    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
    latmmax = eos4p.LambdaOfP123M(lp, g1, g2, g3, mmax)*(mmax*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
    marray = np.linspace(massmin, massmax, nmass)
    mltrue = np.array([[m, eos4p.LambdaOfP123M(lp, g1, g2, g3, m)*(m*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36] for m in marray if m < mmax])
    mltrue = np.append(mltrue, [[mmax, latmmax]], axis=0)
    ax4.plot(mltrue[:, 0], mltrue[:, 1], color='k', ls='-', lw=2.0)
    
    #bounds on masses
    ax4.plot([1.2, 1.2], [0, 12], color='k', ls=':', lw=1.5)
    ax4.plot([1.6, 1.6], [0, 12], color='k', ls=':', lw=1.5)
    
    ax4.set_xlim([0.0, 3.0])
    ax4.set_ylim([0.0, 12.0])
    ax4.set_xlabel(r'$M (M_\odot)$', fontsize=18)
    ax4.set_ylabel(r'$\lambda$ ($10^{36}$ g cm$^2$ s$^2$)', fontsize=18)
    ax4.set_xticklabels(ax4.get_xticks(), fontsize=12)
    ax4.set_yticklabels(ax4.get_yticks(), fontsize=12)


def RAndLambdaPlots(fig, lp, g1, g2, g3, massmin, massmax, nmass, eoschain, conf=[sigma1, sigma2, sigma3], showhistimage=True, \
                    contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Calculate structure quantities (M--R, M--\lambda).
        lp, g1, g2, g3: true values.
        """
    
    ########### 1st plot #############
    
    ax1 = fig.add_subplot(121)
    
    histrange=[0.0, 100.0]
    dy = 0.1
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    # plot confidence bounds
    RadiusConfidenceOfMPlot(ax1, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=showhistimage, \
                            contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    
    # plot true EOS
    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
    ratmmax = eos4p.ROfP123M(lp, g1, g2, g3, eos4p.mmax) # R values past the max mass are set to R(M_max) in the interpolating table
    marray = np.linspace(massmin, massmax, nmass)
    mrtrue = np.array([[m, eos4p.ROfP123M(lp, g1, g2, g3, m)] for m in marray if m < mmax])
    mrtrue = np.append(mrtrue, [[mmax, ratmmax]], axis=0)
    ax1.plot(mrtrue[:, 0], mrtrue[:, 1], color='k', ls='-', lw=2.0)
    
    #bounds on masses
    ax1.plot([1.2, 1.2], [5, 20], color='k', ls=':', lw=1.5)
    ax1.plot([1.6, 1.6], [5, 20], color='k', ls=':', lw=1.5)
    
    ax1.set_xlim([0.0, 3.0])
    ax1.set_ylim([5.0, 20.0])
    ax1.set_xlabel(r'$M (M_\odot)$', fontsize=18)
    ax1.set_ylabel(r'$R$ (km)', fontsize=18)
    ax1.set_xticklabels(ax1.get_xticks(), fontsize=12)
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=12)
    
    ########### 2nd plot #############
    
    ax2 = fig.add_subplot(122)
    
    histrange=[0.0, 20.0]
    dy = 0.1
    bins = int(np.ceil((histrange[1]-histrange[0])/dy))
    # plot confidence bound
    LambdaConfidenceOfMPlot(ax2, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=showhistimage, \
                            contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    
    # plot true EOS
    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
    latmmax = eos4p.LambdaOfP123M(lp, g1, g2, g3, mmax)*(mmax*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
    marray = np.linspace(massmin, massmax, nmass)
    mltrue = np.array([[m, eos4p.LambdaOfP123M(lp, g1, g2, g3, m)*(m*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36] for m in marray if m < mmax])
    mltrue = np.append(mltrue, [[mmax, latmmax]], axis=0)
    ax2.plot(mltrue[:, 0], mltrue[:, 1], color='k', ls='-', lw=2.0)
    
    #bounds on masses
    ax2.plot([1.2, 1.2], [0, 12], color='k', ls=':', lw=1.5)
    ax2.plot([1.6, 1.6], [0, 12], color='k', ls=':', lw=1.5)
    
    ax2.set_xlim([0.0, 3.0])
    ax2.set_ylim([0.0, 12.0])
    ax2.set_xlabel(r'$M (M_\odot)$', fontsize=18)
    ax2.set_ylabel(r'$\lambda$ ($10^{36}$ g cm$^2$ s$^2$)', fontsize=18)
    ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)
    
    fig.subplots_adjust(wspace=0.22)


#################################################################################
############ Make NS structure plots for 4-parameter EOS ########################
#################################################################################
#
#
#def MaxMassConfidencePlot(axes, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False):
#    """
#    Calculate confidence interval for maximum mass and generate histogram.
#    """
#    
#    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
#    # Set R=0 if EOS dosen't allow NS of that mass.
#    maxmass = np.zeros(len(eoschain))
#    for i in range(len(maxmass)):
#        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
#        maxmass[i] = eos4p.MMaxOfP123(lp, g1, g2, g3)
#    
#    mu.ConfidencePlot1DFromChain(axes, maxmass, bins, conf=conf, histrange=histrange, smooth=smooth)
#
#
################# Radius functions ##################
#
#
## It would be better to store R, k2, etc. during the MCMC run
## (I think emcee allows storing of metadata) instead of recalculating it here.
## But, it's not calculated for a uniformly spaced list of masses during the MCMC run,
## so maybe it doesn't matter.
#def RadiusConfidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
#    """
#        Calculate radius confidence interval for given mass.
#        """
#    
#    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
#    # Set R=0 if EOS dosen't allow NS of that mass.
#    radius = np.zeros(len(eoschain))
#    for i in range(len(radius)):
#        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
#        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
#            radius[i] = eos4p.ROfP123M(lp, g1, g2, g3, mass)
#        else:
#            radius[i] = 0.0
#
#    # generate histogram
#    hist, centers = mu.CenteredHistogram1D(radius, bins, histrange=histrange, density=False, smooth=smooth)
#
#    # calculate confidence intervals
#    minmaxlist = np.zeros((len(conf), 2))
#    for n in range(len(conf)):
#        rmin, rmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
#        minmaxlist[n, [0, 1]] = [rmin, rmax]
#
#    ## if rmin corresponds to unstable r=0 NS, find the true rmin
#    #rtruemin = 0.0
#    #if rmin < 2.0: # the rmin value is at the center of the bin, so rmin won't actually be at 0.0
#    #    rtruemin = # the next r-value greater than rmin
#
#    # plot histogram and confidence intervals if requested
#    if showhist:
#        fig = plt.figure(figsize=(4, 3))
#        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#        mu.ConfidencePlot1DFromChain(axes, radius, bins, conf=conf, histrange=histrange, smooth=smooth)
#        axes.set_xlim([-1.0, 25.0])
#
#    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds
#
#
#def RadiusConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False):
#    """
#        Plot R confidence bounds as a function of M.
#        """
#    
#    marray = np.linspace(massmin, massmax, nmass)
#    minmaxlist = []
#    histlist = []
#    for mass in marray:
#        minmax, hist, centers = RadiusConfidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
#        minmaxlist.append(minmax)
#        histlist.append(hist)
#    # centers is the same for all masses so don't make an list of them
#    
#    # plot each confidence interval
#    colorlist = ['red', 'darkgreen', 'blue']
#    for n in range(len(conf)):
#        # truncate the list to m < mMax
#        mrlower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 2.0])
#        mrupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 2.0])
#        axes.plot(mrlower[:, 0], mrlower[:, 1], color=colorlist[n], lw=1.5)
#        axes.plot(mrupper[:, 0], mrupper[:, 1], color=colorlist[n], lw=1.5)
#    
#    # Image of histogram
#    extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
#    vmax = 10*len(eoschain)/bins
#    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
#                     aspect=axes.get_aspect(), norm=Normalize(vmin=0, vmax=vmax))
##    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
##                     aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))
#
#
############## lambda = 2/3G k_2 R^5 (g cm^2 s^2) functions ##############
#
#def LambdaConfidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
#    """
#        Calculate lambda confidence interval for given mass.
#        """
#    
#    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
#    # Set R=0 if EOS dosen't allow NS of that mass.
#    tidal = np.zeros(len(eoschain))
#    for i in range(len(tidal)):
#        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
#        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
#            tidal[i] = eos4p.LambdaOfP123M(lp, g1, g2, g3, mass)*(mass*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
#        else:
#            tidal[i] = 0.0
#    #print(lp, g1, g2, g3, tidal[i])
#    
#    # generate histogram
#    hist, centers = mu.CenteredHistogram1D(tidal, bins, histrange=histrange, density=False, smooth=smooth)
#    
#    # calculate confidence intervals
#    minmaxlist = np.zeros((len(conf), 2))
#    for n in range(len(conf)):
#        tidalmin, tidalmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
#        minmaxlist[n, [0, 1]] = [tidalmin, tidalmax]
#    
#    ## if rmin corresponds to unstable r=0 NS, find the true rmin
#    #rtruemin = 0.0
#    #if rmin < 2.0: # the rmin value is at the center of the bin, so rmin won't actually be at 0.0
#    #    rtruemin = # the next r-value greater than rmin
#    
#    # plot histogram and confidence intervals if requested
#    if showhist:
#        fig = plt.figure(figsize=(4, 3))
#        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#        mu.ConfidencePlot1DFromChain(axes, tidal, bins, conf=conf, histrange=histrange, smooth=smooth)
#        axes.set_xlim([-1.0, 20.0])
#    
#    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds
#
#
#def LambdaConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False):
#    """
#        Plot lambda confidence bounds as a function of M.
#        """
#    
#    marray = np.linspace(massmin, massmax, nmass)
#    minmaxlist = []
#    histlist = []
#    for mass in marray:
#        minmax, hist, centers = LambdaConfidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
#        minmaxlist.append(minmax)
#        histlist.append(hist)
#    # centers is the same for all masses so don't make an list of them
#    
#    # plot each confidence interval
#    colorlist = ['red', 'darkgreen', 'blue']
#    for n in range(len(conf)):
#        # truncate the list to m < mMax
#        mllower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 0.2])
#        mlupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 0.2])
#        axes.plot(mllower[:, 0], mllower[:, 1], color=colorlist[n], lw=1.5)
#        axes.plot(mlupper[:, 0], mlupper[:, 1], color=colorlist[n], lw=1.5)
#    
#    # Image of histogram
#    extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
#    vmax = 10*len(eoschain)/bins
#    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
#                     aspect=axes.get_aspect(), norm=Normalize(vmin=0, vmax=vmax))
##    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
##                     aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))
#
#
############### Love number k_2 functions ###############
#
#def K2Confidence(mass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
#    """
#        Calculate k2 confidence interval for given mass.
#        """
#    
#    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
#    # Set R=0 if EOS dosen't allow NS of that mass.
#    tidal = np.zeros(len(eoschain))
#    for i in range(len(tidal)):
#        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
#        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
#            tidal[i] = eos4p.K2OfP123M(lp, g1, g2, g3, mass)
#        else:
#            tidal[i] = 0.0
#    
#    # generate histogram
#    hist, centers = mu.CenteredHistogram1D(tidal, bins, histrange=histrange, density=False, smooth=smooth)
#    
#    # calculate confidence intervals
#    minmaxlist = np.zeros((len(conf), 2))
#    for n in range(len(conf)):
#        tidalmin, tidalmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
#        minmaxlist[n, [0, 1]] = [tidalmin, tidalmax]
#    
#    # plot histogram and confidence intervals if requested
#    if showhist:
#        fig = plt.figure(figsize=(4, 3))
#        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#        mu.ConfidencePlot1DFromChain(axes, tidal, bins, conf=conf, histrange=histrange, smooth=smooth)
#        axes.set_xlim([-0.01, 0.15])
#    
#    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds
#
#
#def K2ConfidenceOfMPlot(axes, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 25.0], smooth=False, showhist=False):
#    """
#        Plot k2 confidence bounds as a function of M.
#        """
#    
#    marray = np.linspace(massmin, massmax, nmass)
#    minmaxlist = []
#    histlist = []
#    for mass in marray:
#        minmax, hist, centers = K2Confidence(mass, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
#        minmaxlist.append(minmax)
#        histlist.append(hist)
#    # centers is the same for all masses so don't make a list of them
#    
#    # plot each confidence interval
#    colorlist = ['red', 'darkgreen', 'blue']
#    for n in range(len(conf)):
#        # truncate the list to m < mMax
#        mllower = np.array([[marray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 0] > 0.01])
#        mlupper = np.array([[marray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist)) if minmaxlist[i][n, 1] > 0.01])
#        axes.plot(mllower[:, 0], mllower[:, 1], color=colorlist[n], lw=1.5)
#        axes.plot(mlupper[:, 0], mlupper[:, 1], color=colorlist[n], lw=1.5)
#    
#    # Image of histogram
#    extent = [marray[0], marray[-1], histrange[0], histrange[-1]]
#    vmax = 10*len(eoschain)/bins
#    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='Greys', interpolation='none', extent=extent, \
#                     aspect=axes.get_aspect(), norm=Normalize(vmin=1, vmax=vmax, clip=True))
##    im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
##                     aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))
#
#
#def StructurePlots(lp, g1, g2, g3, massmin, massmax, nmass, eoschain, bins, conf=[sigma1, sigma2, sigma3]):
#    """
#        Calculate structure quantities (Mmax, M--R, M--k_2, M--\lambda).
#        lp, g1, g2, g3: true values.
#        """
#    
#    fig = plt.figure(figsize=(8, 6))
#    
#    ########### 1st plot #############
#    
#    ax1 = fig.add_subplot(221)
#    
#    histrange = [0.0, 3.0]
#    MaxMassConfidencePlot(ax1, eoschain, bins, conf=conf, histrange=histrange, smooth=True)
#    truevalue = eos4p.MMaxOfP123(lp, g1, g2, g3)
#    ax1.plot([truevalue, truevalue], [0, 1.1], color='k', ls=':', lw=1.5)
#    ax1.set_xlim([0.0, 3.0])
#    ax1.set_ylim([0.0, 1.1])
#    ax1.set_xlabel(r'$M_{\rm max} (M_\odot)$')
#    
#    ########### 2nd plot #############
#    
#    ax2 = fig.add_subplot(222)
#    
#    histrange=[0.0, 25.0]
#    # plot confidence bounds
#    RadiusConfidenceOfMPlot(ax2, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False)
#    
#    # plot true EOS
#    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
#    ratmmax = eos4p.ROfP123M(lp, g1, g2, g3, eos4p.mmax) # R values past the max mass are set to R(M_max) in the interpolating table
#    marray = np.linspace(massmin, massmax, nmass)
#    mrtrue = np.array([[m, eos4p.ROfP123M(lp, g1, g2, g3, m)] for m in marray if m < mmax])
#    mrtrue = np.append(mrtrue, [[mmax, ratmmax]], axis=0)
#    ax2.plot(mrtrue[:, 0], mrtrue[:, 1], color='k', ls=':', lw=1.5)
#    
#    #bounds on masses
#    ax2.plot([1.2, 1.2], [5, 20], color='k', ls=':', lw=1.5)
#    ax2.plot([1.6, 1.6], [5, 20], color='k', ls=':', lw=1.5)
#    
#    ax2.set_xlim([0.0, 3.0])
#    ax2.set_ylim([5.0, 20.0])
#    ax2.set_xlabel(r'$M (M_\odot)$')
#    ax2.set_ylabel(r'$R$ (km)')
#    
#    ########### 3rd plot #############
#    
#    ax3 = fig.add_subplot(223)
#    
#    histrange=[0.0, 0.15]
#    # plot confidence bounds
#    K2ConfidenceOfMPlot(ax3, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False)
#    
#    # plot true EOS
#    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
#    latmmax = eos4p.K2OfP123M(lp, g1, g2, g3, mmax)
#    marray = np.linspace(massmin, massmax, nmass)
#    mltrue = np.array([[m, eos4p.K2OfP123M(lp, g1, g2, g3, m)] for m in marray if m < mmax])
#    mltrue = np.append(mltrue, [[mmax, latmmax]], axis=0)
#    ax3.plot(mltrue[:, 0], mltrue[:, 1], color='k', ls=':', lw=1.5)
#    
#    #bounds on masses
#    ax3.plot([1.2, 1.2], [0, 0.15], color='k', ls=':', lw=1.5)
#    ax3.plot([1.6, 1.6], [0, 0.15], color='k', ls=':', lw=1.5)
#    
#    ax3.set_xlim([0.0, 3.0])
#    ax3.set_ylim([0.0, 0.15])
#    ax3.set_xlabel(r'$M (M_\odot)$')
#    ax3.set_ylabel(r'$k_2$')
#    
#    ########### 4th plot #############
#    
#    ax4 = fig.add_subplot(224)
#    
#    histrange=[0.0, 20.0]
#    # plot confidence bound
#    LambdaConfidenceOfMPlot(ax4, massmin, massmax, nmass, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False)
#    
#    # plot true EOS
#    mmax = eos4p.MMaxOfP123(lp, g1, g2, g3)
#    latmmax = eos4p.LambdaOfP123M(lp, g1, g2, g3, mmax)*(mmax*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
#    marray = np.linspace(massmin, massmax, nmass)
#    mltrue = np.array([[m, eos4p.LambdaOfP123M(lp, g1, g2, g3, m)*(m*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36] for m in marray if m < mmax])
#    mltrue = np.append(mltrue, [[mmax, latmmax]], axis=0)
#    ax4.plot(mltrue[:, 0], mltrue[:, 1], color='k', ls=':', lw=1.5)
#    
#    #bounds on masses
#    ax4.plot([1.2, 1.2], [0, 12], color='k', ls=':', lw=1.5)
#    ax4.plot([1.6, 1.6], [0, 12], color='k', ls=':', lw=1.5)
#    
#    ax4.set_xlim([0.0, 3.0])
#    ax4.set_ylim([0.0, 12.0])
#    ax4.set_xlabel(r'$M (M_\odot)$')
#    ax4.set_ylabel(r'$\lambda$ ($10^{36}$ g cm$^2$ s$^2$)')


################################################################################
###########                   EOS plots                              ###########
################################################################################


def LogPressureConfidence(rho, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
    """
        Calculate pressure confidence interval for given density.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    logpressure = np.zeros(len(eoschain))
    for i in range(len(logpressure)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        kTab, gammaTab, rhoTab = eos4p.Set4ParameterPiecewisePolytrope(lp, g1, g2, g3)
        logpressure[i] = np.log10(eos4p.POfRhoPiecewisePolytrope(rho, kTab, gammaTab, rhoTab))
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(logpressure, bins, histrange=histrange, density=False, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlist = np.zeros((len(conf), 2))
    for n in range(len(conf)):
        lpmin, lpmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
        minmaxlist[n, [0, 1]] = [lpmin, lpmax]
    
    # plot histogram and confidence intervals if requested
    if showhist:
        fig = plt.figure(figsize=(4, 3))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        mu.ConfidencePlot1DFromChain(axes, logpressure, bins, conf=conf, histrange=histrange, smooth=smooth)
    #axes.set_xlim([32, 36])
    
    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds


def LogPressureConfidenceOfRhoPlot(axes, rhoarray, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[29.0, 40.0], smooth=False, showhist=False, showhistimage=True, \
                                   contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Plot pressure confidence bounds as a function of rho.
        """
    
    logrhoarray = np.log10(np.array(rhoarray))
    minmaxlist = []
    histlist = []
    for logrho in logrhoarray:
        minmax, hist, centers = LogPressureConfidence(10**logrho, eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
        minmaxlist.append(minmax)
        histlist.append(hist)
    # centers is the same for all masses so don't make a list of them
    
    # plot each confidence interval
    for n in range(len(conf)):
        # truncate the list to m < mMax
        lower = np.array([[logrhoarray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist))])
        upper = np.array([[logrhoarray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist))])
        axes.plot(lower[:, 0], lower[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
        axes.plot(upper[:, 0], upper[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
    
    # Image of histogram
    if showhistimage:
        extent = [logrhoarray[0], logrhoarray[-1], histrange[0], histrange[-1]]
        im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
                         aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))


def PressureErrorConfidence(rhotrue, ptrue, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=None, smooth=False, showhist=False):
    """
        Calculate pressure/ptrue confidence interval for given density.
        """
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set R=0 if EOS dosen't allow NS of that mass.
    perror = np.zeros(len(eoschain))
    for i in range(len(perror)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        kTab, gammaTab, rhoTab = eos4p.Set4ParameterPiecewisePolytrope(lp, g1, g2, g3)
        perror[i] = eos4p.POfRhoPiecewisePolytrope(rhotrue, kTab, gammaTab, rhoTab) / ptrue
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(perror, bins, histrange=histrange, density=False, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlist = np.zeros((len(conf), 2))
    for n in range(len(conf)):
        pmin, pmax, level = mu.GetConfidenceInterval1D(hist, centers, conf[n])
        minmaxlist[n, [0, 1]] = [pmin, pmax]
    
    # plot histogram and confidence intervals if requested
    if showhist:
        fig = plt.figure(figsize=(4, 3))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        mu.ConfidencePlot1DFromChain(axes, perror, bins, conf=conf, histrange=histrange, smooth=smooth)
    #axes.set_xlim([32, 36])
    
    return minmaxlist, hist, centers #hist in case you want to make the 2-d histogram instead of just bounds


def PressureErrorOfRhoPlot(axes, rholist, ptruelist, eoschain, bins, conf=[sigma1, sigma2, sigma3], histrange=[0.0, 3.0], smooth=False, showhist=False, showhistimage=True, \
                           contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """
        Plot p(rho)/p_true(rho) confidence bounds as a function of rho.
        rhotrue: densities to plot for true EOS
        ptrue: pressures to plot for true EOS
        """
    
    minmaxlist = []
    histlist = []
    for i in range(len(rholist)):
        minmax, hist, centers = PressureErrorConfidence(rholist[i], ptruelist[i], eoschain, bins, conf=conf, histrange=histrange, smooth=smooth, showhist=showhist)
        minmaxlist.append(minmax)
        histlist.append(hist)
    # centers is the same for all masses so don't make a list of them
    
    logrhoarray = np.log10(np.array(rholist))
    # plot each confidence interval
    for n in range(len(conf)):
        # truncate the list to m < mMax
        lower = np.array([[logrhoarray[i], minmaxlist[i][n, 0]] for i in range(len(minmaxlist))])
        upper = np.array([[logrhoarray[i], minmaxlist[i][n, 1]] for i in range(len(minmaxlist))])
        axes.plot(lower[:, 0], lower[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
        axes.plot(upper[:, 0], upper[:, 1], color=contourcolors[n], ls=contourlinestyles[n], lw=contourlinewidths[n])
    
    # Image of histogram
    if showhistimage:
        extent = [logrhoarray[0], logrhoarray[-1], histrange[0], histrange[-1]]
        im = axes.imshow(np.flipud(np.array(histlist).T), cmap='BuGn', interpolation='none', extent=extent, \
                         aspect=axes.get_aspect(), alpha=1, norm=LogNorm(vmin=0.1, vmax=1000))



def PressureAndErrorOfRhoPlot(fig, rhoarray, ptruearray, eoschain, conf=[sigma1, sigma2, sigma3], \
                              contourcolors=['red', 'darkgreen', 'blue'], contourlinestyles=['-', '-', '-'], contourlinewidths=[2.0, 2.0, 2.0]):
    """Plot p(rho) confidence curves and p(rho)/p(true) confidence curves.
        """
    
    logrhoarray = np.log10(np.array(rhoarray))
    logptruearray = np.log10(np.array(ptruearray))
    
    # Set up plot dimensions
    #    gs = gridspec.GridSpec(2, 1, height_ratios=[1,0.6])
    #    ax1 = fig.add_subplot(gs[0])
    #    ax2 = fig.add_subplot(gs[1])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # stuff for EOS plot
    bins = 500
    histrange = [29.0, 40.0]
    LogPressureConfidenceOfRhoPlot(ax1, rhoarray, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False, showhistimage=False, \
                                   contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    ax1.plot(logrhoarray, logptruearray, color='k', ls='-', lw=3.0)
    ax1.plot([np.log10(2.8e14), np.log10(2.8e14)], histrange, color='k', ls=':', lw=1.5)
    ax1.plot([14.7, 14.7], histrange, color='k', ls=':', lw=1.5)
    ax1.plot([15.0, 15.0], histrange, color='k', ls=':', lw=1.5)
    ax1.set_xlim([logrhoarray[0], logrhoarray[-1]])
    ax1.set_ylim([32, 38])
    ax1.xaxis.set_major_formatter(NullFormatter()) # get rid of x-axis numbers
    ax1.set_ylabel(r'$\log(p)$ (dyn/cm$^2$)', fontsize=18)
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=12)
    
    ax1.text(14.4, 32.4, r'$\rho_{\rm nuc}$', fontsize=18, rotation=90)
    ax1.text(14.63, 33.7, r'$\rho_1 \approx 1.8\rho_{\rm nuc}$', fontsize=18, rotation=90)
    ax1.text(14.93, 33.7, r'$\rho_2 \approx 3.6\rho_{\rm nuc}$', fontsize=18, rotation=90)
    
    #ax1.legend(loc='lower right', fontsize=18)
    
    
    
    # stuff for EOS error plot
    histrange = [0, 10.0]
    bins = 500
    PressureErrorOfRhoPlot(ax2, rhoarray, ptruearray, eoschain, bins, conf=conf, histrange=histrange, smooth=False, showhist=False, showhistimage=False, \
                           contourcolors=contourcolors, contourlinestyles=contourlinestyles, contourlinewidths=contourlinewidths)
    ax2.plot([logrhoarray[0], logrhoarray[-1]], [1.0, 1.0], color='k', ls='-', lw=3.0)
    ax2.plot([np.log10(2.8e14), np.log10(2.8e14)], histrange, color='k', ls=':', lw=1.5)
    ax2.plot([14.7, 14.7], histrange, color='k', ls=':', lw=1.5)
    ax2.plot([15.0, 15.0], histrange, color='k', ls=':', lw=1.5)
    ax2.set_xlim([logrhoarray[0], logrhoarray[-1]])
    ax2.set_ylim([0.0, 3.9])
    ax2.set_xlabel(r'$\log(\rho)$ (g/cm$^3$)', fontsize=18)
    ax2.set_ylabel(r'$p/p_{\rm true}$', fontsize=18)
    ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)
    
    fig.subplots_adjust(hspace=0)


#def PressureAndErrorOfRhoPlot(rhoarray, ptruearray, eoschain, conf=[sigma1, sigma2, sigma3]):
#    """Plot p(rho) confidence curves and p(rho)/p(true) confidence curves.
#        """
#
#    fig = plt.figure(figsize=(6, 8))
#
#    logrhoarray = np.log10(np.array(rhoarray))
#    logptruearray = np.log10(np.array(ptruearray))
#
#    ax1 = fig.add_subplot(211)
#
#    bins = 500
#    histrange = [29.0, 40.0]
#    LogPressureConfidenceOfRhoPlot(ax1, rhoarray, eoschain, bins, conf=conf, histrange=histrange, smooth=True, showhist=False)
#    ax1.plot(logrhoarray, logptruearray, color='k', ls=':', lw=1.5)
#    ax1.plot([np.log10(2.8e14), np.log10(2.8e14)], histrange, color='k', ls=':', lw=1.5)
#    ax1.plot([14.7, 14.7], histrange, color='k', ls=':', lw=1.5)
#    ax1.plot([15.0, 15.0], histrange, color='k', ls=':', lw=1.5)
#    ax1.set_xlim([logrhoarray[0], logrhoarray[-1]])
#    ax1.set_ylim([32, 38])
#    ax1.set_xlabel(r'$\log(\rho)$ (g/cm$^3$)')
#    ax1.set_ylabel(r'$\log(p)$ (dyn/cm$^2$)')
#
#    ax2 = fig.add_subplot(212)
#
#    bins = 100
#    histrange = [0.0, 4.0]
#    PressureErrorOfRhoPlot(ax2, rhoarray, ptruearray, eoschain, bins, conf=conf, histrange=histrange, smooth=False, showhist=False)
#    ax2.plot([np.log10(2.8e14), np.log10(2.8e14)], histrange, color='k', ls=':', lw=1.5)
#    ax2.plot([14.7, 14.7], histrange, color='k', ls=':', lw=1.5)
#    ax2.plot([15.0, 15.0], histrange, color='k', ls=':', lw=1.5)
#    ax2.plot([logrhoarray[0], logrhoarray[-1]], [1.0, 1.0], color='k', ls=':', lw=1.5)#    ax2.set_xlim([logrhoarray[0], logrhoarray[-1]])
#    ax2.set_ylim(histrange)
#    ax2.set_xlabel(r'$\log(\rho)$ (g/cm$^3$)')
#    ax2.set_ylabel(r'$p/p_{\rm true}$')

