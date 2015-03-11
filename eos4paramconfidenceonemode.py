import numpy as np
#import pickle
import multiprocessing
from functools import partial
import eos4parameterpolytrope as eos4p
import mcmcutilitiesnoplot as mu
#import mcmcutilities as mu
import scipy.interpolate as interpolate
import scipy.stats as stats
import binaryutilities as binary

sigma1 = 0.68268949 # ~317 points out of 1000 are outside
sigma2 = 0.95449974 # ~46 points out of 1000 are outside
sigma3 = 0.99730024 # ~3 points out of 1000 are outside


################# Load set of emcee runs and test convergence. #################

def load_multiple_emcee_runs(file_prefix, file_run_list, file_suffix, nBurn=0, dSteps=1, flatten=False):
    """
    Loads multiple emcee runs.
    
    Args:
        file_prefix (str): File name before the number string
        file_run_list (list of str): List of strings identifying the individual runs
        file_suffix (str): File name after the number string
        flatten (bool, optional): Flatten the separate runs into a single chain (default False)
        
    Returns:
        chainarray: 3d array if not flattened, 2d array if flattened
    """
    
    # Load all the runs into an array of shape (nruns, nsteps, nparams)
    chainarray = np.array([mu.flatten_emcee_chain(np.load(file_prefix + run + file_suffix), nBurn=nBurn, dSteps=dSteps) for run in file_run_list])
    
    # Print the gelman_rubin convergence r statistic for each parameter for fun
    print('r = ', gelman_rubin(chainarray))
    
    if flatten is False:
        return chainarray
    else:
        # Flatten the runs into a single chain
        nruns, nsteps, nparams = chainarray.shape
        return np.reshape(chainarray, (nruns*nsteps, nparams))


def gelman_rubin(mcmc_chains_array):
    """
        Args:
        mcmc_chains_array: 3d array of shape (mseq, niter, nparam)
        mseq is the number of MCMC chains
        niter is the number of iterations in each chain
        nparam is the number of parameters in each chain
        
        Returns:
        R: Gelman-Rubin statistic for each of the nparam parameters.
        R approaches 1 from above as the chains converge.
        Values of R <~ 1.1 are probably? good.
        """
    
    # number of parameters
    nparam = len(mcmc_chains_array[0, 0, :])
    
    return np.array([gelman_rubin_param(mcmc_chains_array[:, :, n]) for n in range(nparam)])


def gelman_rubin_param(paramarray):
    """
        Args:
        paramarray: 2d array of samples of the parameter param with shape (mseq, niter)
        mseq is the number of MCMC chains
        niter is the number of iterations in each chain
        
        Returns:
        R: Gelman-Rubin statistic.
        R approaches 1 from above as the chains converge.
        Values of R <~ 1.1 are probably? good.
        """
    
    # mseq MCMC chains/runs/sequences
    # niter iterations per MCMC chain
    mseq, niter = paramarray.shape
    
    # Average value of the parameter for each of the mseq sequences (MCMC runs)
    # Each sequence has niter points
    # Mean is calculated for each value in the first index (axis=0), summing over the second index (axis=1).
    seq_mean = np.mean(paramarray, axis=1)
    
    # Variance of the mean parameter values for the mseq sequences (with mseq-1 degrees of freedom)
    # This is a measure of how much the mean parameter value varies from one sequence to another
    Bbyn = np.var(seq_mean, ddof=1)
    
    # Variance of the parameter param (with niter-1 degrees of freedom)
    # for each of the mseq sequences
    # This is the uncertainty in the parameter as found by each sequence
    seqvar = np.var(paramarray, axis=1, ddof=1)
    
    # Average of the variances for the mseq MCMC chains
    # This is the average of the uncertainty in the parameter found by each sequence
    W = np.mean(seqvar)
    
    # Estimated target mean
    # Mean of the mseq * niter points
    # np.mean will flatten a 2d array before taking the mean unless axis is specified
    targetmean = np.mean(paramarray)
    
    # Estimated target variance
    targetvar = (niter-1.0)/float(niter) * W + Bbyn
    
    # Estimate the "scale reduction factor"
    # Standard deviation of the targetvar estimate / average of the standard deviation of the parameter for each chain
    # How much you expect the estimate of the uncertainty to decrease if you let the chain go on forever
    R = np.sqrt(targetvar/W)
    
    return R


def interpolate_p_of_rho(rparray):
    """
        Do linear interpolation of ln(p)(ln(rho)), then exponentiate to get p(rho) = e^{ln(p)(ln(rho))}.
        This is guaranteed to be monotonic if the rptable is monotonic.
        
        Args:
        rparray (array): [[rho0, p0], [rho1, p1], ...]
        
        Returns:
        pofr: pressure(rho) function object
        """
    
    # To do: Add assertions that the table is positive and monotonic
    
    lograrray = np.log(rparray[:, 0])
    logparray = np.log(rparray[:, 1])
    
    # Linear interpolation is guaranteed to be monotonic
    # Also interp1d sucks for any other kind of interpolation
    logpoflogr = interpolate.interp1d(lograrray, logparray, kind=1)
    
    # Generate function object
    # Account for points out of range because interp1d doesn't extrapolate
    def pofr(r):
        lnr = np.log(r)
        if lnr <= lograrray[0]:
            return rparray[0, 1]
        elif lnr >= lograrray[-1]:
            return rparray[-1, 1]
        else:
            return np.exp(logpoflogr(np.log(r)))
    
    return pofr


def divide_tabulated_eos(rhop1array, rhop2array, nrho=100):
    """
        Evaluate p1/p2 from two tabulated EOSs.
        
        Args:
        rhop1array (array): Numerator with format np.array([r0, p0], [r1, p1], ...])
        rhop2array (array): Denomenator
        nrho (int, optional): number of points to interpolate table at
        
        Returns:
        array of np.array([log(rho), p2/p1], ...])
        """
    
    # Interpolate tables
    p1ofrho = interpolate_p_of_rho(rhop1array)
    p2ofrho = interpolate_p_of_rho(rhop2array)
    
    # Density range common to both tables
    rhomin = max(rhop1array[0, 0], rhop2array[0, 0])
    rhomax = min(rhop1array[-1, 0], rhop2array[-1, 0])
    
    # log-spaced densities
    logrhoarray = np.linspace(np.log10(rhomin), np.log10(rhomax), nrho)
    rhoarray = 10**logrhoarray
    
    return np.array([[rho, p1ofrho(rho)/p2ofrho(rho)] for rho in rhoarray])


############### Functions for generating confidence regions ################

def get_confidence_interval_single_mode_plus_delta(hist, xcenters, conf, lower_bound=None, upper_bound=None, dx=None):
    """Generates a confidence interval.
        Assumes the histogram contains a single mode with compact support in the region (lower_bound, upper_bound),
        plus 0 or more delta functions (infinite but with finite area) outside this region.
        
        The interval is chosen to be the smallest interval s.t.
        [area of interval] = conf * [total area] - [weight of delta functions outside (lower_bound, upper_bound)]
        
        The histogram can have any overall normalization constant [total area].
        
        By default (lower_bound, upper_bound) is the bounds of the histogram: (xcenters[0], xcenters[-1]).
        """
    
    # Set grid size if not done explicitly to 0.1 times the current bin width
    if dx is None: dx = 0.1*(xcenters[1] - xcenters[0])
    # Number of grid points for the entire domain
    ngrid = (xcenters[-1] - xcenters[0]) / dx + 1.0
    # Interpolate the histogram to make more finely spaced grid
    f = interpolate.interp1d(xcenters, hist, kind='linear')
    #f = interpolate.interp1d(xcenters, hist, kind='cubic')
    xcenters = np.linspace(xcenters[0], xcenters[-1], ngrid)
    # Correct the interpolation in case it gives something less than 0.0
    hist = f(xcenters)
    
    # Get default values for (lower_bound, upper_bound) interval from the bounds on the histogram
    if lower_bound is None: lower_bound = xcenters[0]
    if upper_bound is None: upper_bound = xcenters[-1]
    
    # index bounds for middle part
    # will be just inside (or exactly) the region (lower_bound, upper_bound)
    ilow = 0
    while xcenters[ilow] < lower_bound: ilow += 1
    ihigh = len(xcenters) - 1
    while xcenters[ihigh] > upper_bound: ihigh -= 1
    
    # Calculate weight below and above the range (lower_bound, upper bound)
    sum_total = np.sum(hist)
    sum_lower = np.sum(hist[0:ilow]) # this does not include ilow
    sum_upper = np.sum(hist[ihigh+1:len(xcenters)-1]) # this does not include ihigh
    #print(hist)
    #print(hist[ihigh+1:len(xcenters)-1])
    #print(ihigh+1, len(xcenters)-1, sum_upper)
    
    # Exit the function if you've already accumulated conf outside the range (lower_bound, upper bound)
    # This also takes care of the case where all the values are 0.0 because
    # in this case sum_lower = sum_upper = sum_total = 0.0.
    if sum_lower + sum_upper >= conf * sum_total:
        # return some reasonable values
        xmin, xmax, level = xcenters[0], xcenters[0], 0.0
        return xmin, xmax, level
    
    # sort the bins in the range (lower_bound, upper bound) from smallest to greatest
    sort_mid = np.sort(hist[ilow:ihigh+1]) # this does include ilow and ihigh, but not ihigh+1
    
    # sum all the bins in the middle
    sum_mid = np.sum(sort_mid)
    
    # start at largest bin then add the bins until you get to frac fraction of the total
    isum = len(sort_mid) - 1
    sum_interval = 0
    # isum can get to zero in the special case where all the bins are zero,
    # or where you are using a really large confidence (e.g. 0.99999)
    # which is essentially the whole range
    while (sum_interval < sum_total * conf - sum_lower - sum_upper) and (isum > 0):
        sum_interval += sort_mid[isum]
        isum -= 1
    level = sort_mid[isum]
    
    # Find the lower value of the confidence interval.
    # You already have xcenters(ilow) set to just above or equal to lower_bound.
    while hist[ilow] < level:
        ilow += 1
    xmin = xcenters[ilow]
    
    # Find upper value of confidence interval.
    # Start summing from xmin until you reach the necessary confidence.
    # This method is robust against little jitters in an otherwise single-mode histogram.
    ihigh = ilow
    sum_interval = 0.0
    while (sum_interval < sum_total * conf - sum_lower - sum_upper):
        sum_interval += hist[ihigh]
        ihigh += 1
    xmax = xcenters[ihigh]
    
    #print(np.array([
    #                float(sum_lower), float(sum_mid), float(sum_upper), float(sum_interval),
    #                float(sum_lower+sum_mid+sum_upper), float(sum_lower+sum_interval+sum_upper)
    #                ]) / sum_total)

    return xmin, xmax, level


def get_confidence_interval_from_samples(xvalues, conf, lower_bound=None, upper_bound=None):
    """
        Generates a confidence interval within the region (lower_bound, upper_bound).
        If points are outside this region, they are counted towards the confidence.
        This is equivalent to assuming the PDF has 0 or more delta functions outside this region.
        
        The interval is the smallest one containing the fraction conf of points.
        
        Args:
        xvalues (array): data samples for parameter x
        conf (float): Confidence in range (0.0 1.0)
        lower_bound (float, optional): Defaults to smallest value
        upper_bound (float, optional): Defaults to largest value
        
        Returns:
        xmin, xmax, level (level is just 0.0)
        """
    
    npoints = len(xvalues)
    
    # sort the points from smallest to greatest
    xsort = np.sort(xvalues)
    
    # Get default values for (lower_bound, upper_bound) interval from the bounds on the histogram
    if lower_bound is None: lower_bound = xsort[0]
    if upper_bound is None: upper_bound = xsort[-1]
    #print("(lower_bound, upper_bound) = (", lower_bound, upper_bound, ")")
    
    # index bounds for middle part
    # will be just inside (or exactly) the region (lower_bound, upper_bound)
    ilow = 0
    while (xsort[ilow] < lower_bound) and (ilow < npoints-1): ilow += 1
    ihigh = npoints - 1
    while (xsort[ihigh] > upper_bound) and (ihigh > 0): ihigh -= 1
    
    # Number of points below and above the range (lower_bound, upper bound)
    nlower = ilow # points xsort[0] to xsort[ilow-1]
    nupper =  npoints-1-ihigh # points xsort[ihigh+1] to xsort[npoints-1]
    
    # Number of points needed in region (lower_bound, upper_bound)
    nmid = int(np.ceil(conf*npoints - nlower - nupper))
    
    # Exit the function if you've already accumulated conf outside the range (lower_bound, upper bound)
    if nmid < 0:
        # return some reasonable values
        xmin, xmax, level = xsort[0], xsort[0], 0.0
        return xmin, xmax, level
    
    # Find minimum interval containing nmid points
    dxmin = xsort[ihigh] - xsort[ilow]
    iminlow = ilow
    for i in range(ilow, ihigh-nmid+1):
        dx = xsort[i+nmid]-xsort[i]
        if dx < dxmin:
            dxmin = dx
            iminlow = i
    
    # level has to be some value so call it 0.0
    xmin, xmax, level = xsort[iminlow], xsort[iminlow+nmid], 0.0
    
    return xmin, xmax, level

def get_confidence_interval_from_samples_restrict_one_mode(xvalues, conf, lower_bound=None, upper_bound=None):
    """
    Generates a confidence interval within the region (lower_bound, upper_bound).
    If points are outside this region, ignore them and only calculate confidence for points in the region (lower_bound, upper_bound).
    This is different from get_confidence_interval_from_samples.
        
    The interval is the smallest one containing the fraction conf of points.
    
    Args:
        xvalues (array): data samples for parameter x
        conf (float): Confidence in range (0.0 1.0)
        lower_bound (float, optional): Defaults to smallest value
        upper_bound (float, optional): Defaults to largest value
    
    Returns:
        xmin, xmax, level (level is just 0.0)
    """
    
    npoints = len(xvalues)
    
    # sort the points from smallest to greatest
    xsort = np.sort(xvalues)
    
    # Get default values for (lower_bound, upper_bound) interval from the bounds on the histogram
    if lower_bound is None: lower_bound = xsort[0]
    if upper_bound is None: upper_bound = xsort[-1]
    #print("(lower_bound, upper_bound) = (", lower_bound, upper_bound, ")")
    
    # index bounds for middle part
    # will be just inside (or exactly) the region (lower_bound, upper_bound)
    ilow = 0
    while (xsort[ilow] < lower_bound) and (ilow < npoints-1): ilow += 1
    ihigh = npoints - 1
    while (xsort[ihigh] > upper_bound) and (ihigh > ilow): ihigh -= 1
    
    # Exit the function if there are no points inside the range (lower_bound, upper bound)
    if ihigh - ilow <= 0:
        # return some reasonable values
        xmin, xmax, level = xsort[0], xsort[0], 0.0
        return xmin, xmax, level
    
    # Only the points in the mode
    xsortmode = xsort[ilow:ihigh]
    
    # Number of points in middle region that constitutes conf fraction of points in region (lower_bound, upper_bound)
    nmid = int(np.floor(conf*len(xsortmode)))
    
    # Find minimum interval containing nmid points
    dxmin = xsort[ihigh] - xsort[ilow]
    iminlow = ilow
    for i in range(ilow, ihigh-nmid+1):
        dx = xsort[i+nmid]-xsort[i]
        if dx < dxmin:
            dxmin = dx
            iminlow = i
    
    # level has to be some value so call it 0.0
    xmin, xmax, level = xsort[iminlow], xsort[iminlow+nmid], 0.0
    
    return xmin, xmax, level


########## Confidence regions for log(pressure) and error in pressure ##########

def log_pressure_confidence(rho, eoschain, conf=[sigma1, sigma2, sigma3], method='points', dx=0.1, smooth=False, showhist=False):
    """
        Calculate log10(p) confidence interval for given density rho.
        
        Args:
        rho (float)
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        method (str): 'points' looks for minimum spacing containing conf fraction of points.
        'histogram' generates a histogram then sums bins.
        dx (float, optional): Bin spacing. Defaults to dlog(p) = 0.1.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to False.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        list: List of [logplow, logphigh] pairs for each confidence in conf.
        array: histogram array
        array: Centers for the histogram (spacing dx).
        """
    
    # Calculate log10(p) for each EOS in chain
    logp = np.zeros(len(eoschain))
    for i in range(len(logp)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        k_tab, gamma_tab, rho_tab = eos4p.Set4ParameterPiecewisePolytrope(lp, g1, g2, g3)
        logp[i] = np.log10(eos4p.POfRhoPiecewisePolytrope(rho, k_tab, gamma_tab, rho_tab))

    # Get histrange and number of bins
    # Put a decent amount of buffer on the histrange for when smooth=True
    histrange = [np.min(logp)-5.0*dx, np.max(logp)+5.0*dx]
    bins = int(np.ceil((histrange[1]-histrange[0])/dx))

    # generate histogram
    hist, centers = mu.CenteredHistogram1D(logp, bins, histrange=histrange, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlevel = np.zeros((len(conf), 3))
    for n in range(len(conf)):
        pmin, pmax, level = get_confidence_interval_from_samples(logp, conf[n])
        #if method is 'points':
        #    pmin, pmax, level = get_confidence_interval_from_samples(logp, conf[n])
        #else:
        #    pmin, pmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n])
        minmaxlevel[n] = [pmin, pmax, level]

    # plot histogram and confidence intervals if requested
    if showhist:
        ymax = np.max(hist)
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(centers, hist, c='k', ls='-', lw=2.0)
        axes.set_xlim(histrange)
        axes.set_ylim([0.0, ymax])
        color = ['red', 'blue', 'darkgreen']
        ls = [':', '--', '-']
        for n in range(len(minmaxlevel)):
            #axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            #axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)

    #return minmaxlevel[:, [0, 1]], hist, centers
    return minmaxlevel[:, [0, 1]]


def pressure_error_confidence(rho, ptrue, eoschain, conf=[sigma1, sigma2, sigma3], method='points', dx=0.1, smooth=False, showhist=False):
    """
        Calculate p/ptrue confidence interval for given density rho.
        
        Args:
        rho (float)
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to dlog(p) = 0.1.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to False.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        list: List of [(p/ptrue)_low, (p/ptrue)_high] pairs for each confidence in conf.
        array: histogram array
        array: Centers for the histogram (spacing dx).
        """
    
    # Calculate log10(p) for each EOS in chain
    perror = np.zeros(len(eoschain))
    for i in range(len(perror)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        k_tab, gamma_tab, rho_tab = eos4p.Set4ParameterPiecewisePolytrope(lp, g1, g2, g3)
        perror[i] = eos4p.POfRhoPiecewisePolytrope(rho, k_tab, gamma_tab, rho_tab) / ptrue
    
    # Get histrange and number of bins
    # Put a decent amount of buffer on the histrange for when smooth=True
    histrange = [np.min(perror)-5.0*dx, np.max(perror)+5.0*dx]
    bins = int(np.ceil((histrange[1]-histrange[0])/dx))
    
    # generate histogram
    hist, centers = mu.CenteredHistogram1D(perror, bins, histrange=histrange, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlevel = np.zeros((len(conf), 3))
    for n in range(len(conf)):
        pmin, pmax, level = get_confidence_interval_from_samples(perror, conf[n])
        #if method is 'points':
        #    print(method)
        #    pmin, pmax, level = get_confidence_interval_from_samples(perror, conf[n])
        #else:
        #    pmin, pmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n])
        minmaxlevel[n] = [pmin, pmax, level]

    # plot histogram and confidence intervals if requested
    if showhist:
        ymax = np.max(hist)
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(centers, hist, c='k', ls='-', lw=2.0)
        axes.set_xlim(histrange)
        axes.set_ylim([0.0, ymax])
        color = ['red', 'blue', 'darkgreen']
        ls = [':', '--', '-']
        for n in range(len(minmaxlevel)):
            #axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            #axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)

    #return minmaxlevel[:, [0, 1]], hist, centers
    return minmaxlevel[:, [0, 1]]


def log_pressure_confidence_of_log_rho(rhomin, rhomax, nrho, eoschain, conf=[sigma1, sigma2, sigma3], method='points', dx=0.1, smooth=True, showhist=False):
    """
        Get log10(p) confidence bounds as a function of density rho. Logarithmically space the densities.
        
        Args:
        rhomin (float)
        rhomax (float)
        nrho (int): number of densities logarithmically spaced from rhomin to rhomax
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to dlog(p) = 0.1.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to True.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        array: np.array([[log10(rho), lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...], ...])
        """
    
    # Number of processes defaults to number of cpus
    pool = multiprocessing.Pool()
    
    logrhoarray = np.linspace(np.log10(rhomin), np.log10(rhomax), nrho)
    rhoarray = 10.0**logrhoarray
    
    # For some reason you have to specify the name of the argument eoschain (even though it's not an optional argument)
    # Also the first argument (rho) is the only one that you can allow to vary in the new partial function perrorofrho
    logpofrho = partial(log_pressure_confidence, eoschain=eoschain, conf=conf, method=method, dx=dx, smooth=smooth, showhist=showhist)
    minmax = pool.map(logpofrho, rhoarray)
    pool.terminate()
    
    bounds = np.array([np.concatenate((np.array([logrhoarray[n]]), minmax[n].flatten())) for n in range(len(minmax))])
    
    return bounds


def pressure_error_confidence_helper(rho, rhoptruearray, eoschain, conf=[sigma1, sigma2, sigma3], method='points', dx=0.1, smooth=False, showhist=False):
    """
        Calculate p/ptrue confidence interval for given density rho.
        This is a helper function to so that all the rho dependence is in the first argument.
        This will allow the multiprocessing.pool.map function to work properly.
        
        Args:
        rho (float)
        rhoptruearray (array): Format [[rho0, ptrue0], [rho1, ptrue1], ...]
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to dlog(p) = 0.1.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to False.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        minmax (list): lower and upper confidence limints [(p/ptrue)_low, (p/ptrue)_high].
        """
    
    # Interpolate true EOS table then make an array
    ptrueofrho = interpolate_p_of_rho(rhoptruearray)
    ptrue = ptrueofrho(rho)
    
    minmax = pressure_error_confidence(rho, ptrue, eoschain, conf=conf, method=method, dx=dx, smooth=smooth, showhist=showhist)
    
    return minmax


def pressure_error_confidence_of_log_rho(rhomin, rhomax, nrho, rhoptruearray, eoschain, conf=[sigma1, sigma2, sigma3], method='points', dx=0.1, smooth=True, showhist=False):
    """
        Get p(rho)/p_true(rho) confidence bounds as a function of density rho. Logarithmically space the densities.
        
        Args:
        rhomin (float)
        rhomax (float)
        nrho (int): number of densities logarithmically spaced from rhomin to rhomax
        rhoptruearray (array): Format [[rho0, ptrue0], [rho1, ptrue1], ...]
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to d(p/ptrue) = 0.1.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to True.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        bounds (array): np.array([[log10(rho), lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...], ...])
        """
    
    rhomin = max(rhomin, rhoptruearray[0, 0])
    rhomax = min(rhomax, rhoptruearray[-1, 0])
    assert (rhomin < rhomax), 'The following must be true: rhomin < rhomax and rhoptruearray[0, 0] < rhoptruearray[-1, 0]'
    
    # Number of processes defaults to number of cpus
    pool = multiprocessing.Pool()
    
    logrhoarray = np.linspace(np.log10(rhomin), np.log10(rhomax), nrho)
    rhoarray = 10.0**logrhoarray
    
    # For some reason you have to specify the name of the arguments rhoptruearray and eoschain (even though they're not optional arguments)
    # Also the first argument (rho) is the only one that you can allow to vary in the new partial function perrorofrho
    perrorofrho = partial(pressure_error_confidence_helper, rhoptruearray=rhoptruearray, eoschain=eoschain, conf=conf, method=method, dx=dx, smooth=smooth, showhist=showhist)
    minmax = pool.map(perrorofrho, rhoarray)
    pool.terminate()
    
    #print(logrhoarray, minmax)
    
    bounds = np.array([np.concatenate((np.array([logrhoarray[n]]), minmax[n].flatten())) for n in range(len(minmax))])
    
    return bounds


########## Confidence regions for radius R and tidal parameter lambda ##########

def radius_confidence(mass, eoschain, conf=[sigma1, sigma2, sigma3], dx=0.1, smooth=False, showhist=False):
    """
        Calculate radius confidence interval for given mass.
        
        When some of the MCMC realizations result in BH instead of NS, set the radius to radius_BH.
        The distribution then becomes bimodal with a delta function at radius_BH.
        
        Args:
        mass (float)
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to 0.1km.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to False.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        list: List of [rlow, rhigh] pairs for each confidence in conf.
        array: histogram array
        array: Centers for the histogram (spacing dx).
        """
    
    # Sufficiently far away from 0.0 for when smooth=True smooths out the delta function
    radius_BH = -5.0*dx
    
    # Calculate radius and check to make sure the mass you are plotting is allowed by that EOS
    # Set radius = radius_BH if EOS dosen't allow NS of that mass.
    radius = np.zeros(len(eoschain))
    for i in range(len(radius)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
            radius[i] = eos4p.ROfP123M(lp, g1, g2, g3, mass)
        else:
            radius[i] = radius_BH

    # Get histrange and number of bins
    # Put a decent amount of buffer on the histrange for when smooth=True
    # If all stars are BHs at current mass, then set upper bound to 10km
    upper = max(np.max(radius)+5.0*dx, 10.0)
    # Cap range at 500km so there aren't a rediculous number of bins
    upper = min(upper, 500)
    histrange = [radius_BH-5.0*dx, upper]
    bins = int(np.ceil((histrange[1]-histrange[0])/dx))

    # generate histogram
    hist, centers = mu.CenteredHistogram1D(radius, bins, histrange=histrange, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlevel = np.zeros((len(conf), 3))
    for n in range(len(conf)):
        rmin, rmax, level = get_confidence_interval_from_samples_restrict_one_mode(radius, conf[n], lower_bound=0.0)
        #if method is 'points':
        #    rmin, rmax, level = get_confidence_interval_from_samples(radius, conf[n], lower_bound=0.0)
        #else:
        #    rmin, rmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n], lower_bound=0.0, upper_bound=histrange[1])
        minmaxlevel[n] = [rmin, rmax, level]

    ## calculate confidence intervals
    #minmaxlevel = np.zeros((len(conf), 3))
    #for n in range(len(conf)):
    #    if method == 'points':
    #        rmin, rmax, level = get_confidence_interval_from_samples(radius, conf[n], lower_bound=0.0)
    #    elif method == 'restrictmode':
    #        rmin, rmax, level = get_confidence_interval_from_samples_restrict_one_mode(radius, conf[n], lower_bound=0.0)
    #    else:
    #        rmin, rmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n], lower_bound=0.0, upper_bound=histrange[1])
    #    minmaxlevel[n] = [rmin, rmax, level]

    # plot histogram and confidence intervals if requested
    if showhist:
        ymax = np.max(np.array([hist[i] for i in range(len(hist)) if centers[i] > 0.0]))
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(centers, hist, c='k', ls='-', lw=2.0)
        axes.set_xlim(histrange)
        axes.set_ylim([0.0, ymax])
        axes.plot([0.0, 0.0], [0.0, max(1.0, ymax)], color='r', ls=':', lw=1.5)
        axes.plot([histrange[1], histrange[1]], [0.0, ymax], color='r', ls=':', lw=1.5)
        color = ['red', 'blue', 'darkgreen']
        ls = [':', '--', '-']
        for n in range(len(minmaxlevel)):
            #axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            #axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)

    #return minmaxlevel[:, [0, 1]], hist, centers
    return minmaxlevel[:, [0, 1]]


def lambda_confidence(mass, eoschain, conf=[sigma1, sigma2, sigma3], dx=0.1, smooth=False, showhist=False):
    """
        Calculate lambda confidence interval for given mass.
        
        When some of the MCMC realizations result in BH instead of NS, set the radius to radius_BH.
        The distribution then becomes bimodal with a delta function at radius_BH.
        
        Args:
        mass (float)
        eoschain (array): Chain of EOS parameters from MCMC simulation.
        conf (list, optional): List of confidences between (0.0, 1.0).
        dx (float, optional): Bin spacing. Defaults to 0.1km.
        smooth (bool, optional): Use Gaussian smoothing with standard deviation
        of 0.75 times bin width if True. Defaults to False.
        showhist (bool, optional): Generate histogram with confidence intervals drawn if True.
        Defaults to False.
        
        Returns:
        list: List of [rlow, rhigh] pairs for each confidence in conf.
        array: histogram array
        array: Centers for the histogram (spacing dx).
        """
    
    # Sufficiently far away from 0.0 for when smooth=True smooths out the delta function
    tidal_BH = -5.0*dx
    
    # Calculate tidal parameter and check to make sure the mass you are plotting is allowed by that EOS
    # Set tidal = tidal_BH if EOS dosen't allow NS of that mass.
    tidal = np.zeros(len(eoschain))
    for i in range(len(tidal)):
        lp, g1, g2, g3 = eoschain[i, 0], eoschain[i, 1], eoschain[i, 2], eoschain[i, 3]
        if eos4p.MMaxOfP123(lp, g1, g2, g3) >= mass:
            tidal[i] = eos4p.LambdaOfP123M(lp, g1, g2, g3, mass)*(mass*binary.MSUN_CGS*binary.G_CGS/binary.C_CGS**2)**5/binary.G_CGS/1.0e36
        else:
            tidal[i] = tidal_BH

    # Get histrange and number of bins
    # Put a decent amount of buffer on the histrange for when smooth=True
    # If all stars are BHs at current mass, then set upper bound to 10km
    upper = max(np.max(tidal)+5.0*dx, 10.0)
    # Cap range at 500 so there aren't a rediculous number of bins
    upper = min(upper, 500)
    histrange = [tidal_BH-5.0*dx, upper]
    bins = int(np.ceil((histrange[1]-histrange[0])/dx))

    # generate histogram
    hist, centers = mu.CenteredHistogram1D(tidal, bins, histrange=histrange, smooth=smooth)
    
    # calculate confidence intervals
    minmaxlevel = np.zeros((len(conf), 3))
    for n in range(len(conf)):
        lmin, lmax, level = get_confidence_interval_from_samples_restrict_one_mode(tidal, conf[n], lower_bound=0.0)
        #if method is 'points':
        #    lmin, lmax, level = get_confidence_interval_from_samples(tidal, conf[n], lower_bound=0.0)
        #else:
        #    lmin, lmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n], lower_bound=0.0, upper_bound=histrange[1])
        minmaxlevel[n] = [lmin, lmax, level]

    ## calculate confidence intervals
    #minmaxlevel = np.zeros((len(conf), 3))
    #for n in range(len(conf)):
    #    if method == 'points':
    #        lmin, lmax, level = get_confidence_interval_from_samples(tidal, conf[n], lower_bound=0.0)
    #    elif method == 'restrictmode':
    #        lmin, lmax, level = get_confidence_interval_from_samples_restrict_one_mode(tidal, conf[n], lower_bound=0.0)
    #    else:
    #        lmin, lmax, level = get_confidence_interval_single_mode_plus_delta(hist, centers, conf[n], lower_bound=0.0, upper_bound=histrange[1])
    #    minmaxlevel[n] = [lmin, lmax, level]

    # plot histogram and confidence intervals if requested
    if showhist:
        ymax = np.max(np.array([hist[i] for i in range(len(hist)) if centers[i] > 0.0]))
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(centers, hist, c='k', ls='-', lw=2.0)
        axes.set_xlim(histrange)
        axes.set_ylim([0.0, max(1.0, ymax)])
        axes.plot([0.0, 0.0], [0.0, ymax], color='r', ls=':', lw=1.5)
        axes.plot([histrange[1], histrange[1]], [0.0, ymax], color='r', ls=':', lw=1.5)
        color = ['red', 'blue', 'darkgreen']
        ls = [':', '--', '-']
        for n in range(len(minmaxlevel)):
            #axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            #axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, minmaxlevel[n, 2]], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 0], minmaxlevel[n, 0]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)
            axes.plot([minmaxlevel[n, 1], minmaxlevel[n, 1]], [0, ymax], color=color[n%3], ls=ls[n%3], lw=2.0)

    #return minmaxlevel[:, [0, 1]], hist, centers
    return minmaxlevel[:, [0, 1]]


def confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=[sigma1, sigma2, sigma3], dx=0.1, smooth=True, showhist=False):
    """
        Get confidence bounds as a function of mass.
        
        Args:
        quantity (str): 'radius' or 'lambda'
        
        Returns:
        array: np.array([[mass, lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...], ...])
        """
    
    # Number of processes defaults to number of cpus
    pool = multiprocessing.Pool()
    
    # Array of masses to calculate bounds for
    marray = np.linspace(massmin, massmax, nmass)
    
    # Generate list with lower and upper bounds for each confidence interval
    # Columns are: mass, lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...
    if quantity == 'radius':
        # For some reason you have to specify the name of the argument eoschain (even though eoschain is not an optional argument)
        # Also the first argument (m) is the only one that you can allow to vary in the new partial function rofm
        rofm = partial(radius_confidence, eoschain=eoschain, conf=conf, dx=dx, smooth=smooth, showhist=showhist)
        minmax = pool.map(rofm, marray)
    elif quantity == 'lambda':
        lofm = partial(lambda_confidence, eoschain=eoschain, conf=conf, dx=dx, smooth=smooth, showhist=showhist)
        minmax = pool.map(lofm, marray)
    else:
        print("quantity must be 'radius' or 'lambda'")
        return None
    
    bounds = np.array([np.concatenate((np.array([marray[n]]), minmax[n].flatten())) for n in range(len(minmax))])
    
    return bounds


################ Plotting confidence regions ###################

def plot_confidence_of_log_rho(axes, bounds, colors=['red', 'blue', 'darkgreen'], linestyles=['-', '-', '-'], linewidths=[2.0, 2.0, 2.0], alpha=0.2, label=None):
    """
        Make a filled plot of the confidence regions from the bounds table.
        """
    
    assert (
            len(colors) == len(linestyles) == len(linewidths) == (len(bounds[0])-1)/2
            ), 'confidence bounds, colors, linestyles, linewidths must have consistent length.'
    
    # plot each confidence interval
    # plot last (largest) confidence interval first, then overlap the smaller ones
    for n in reversed(range(len(colors))):
        x = bounds[:, 0]
        ylower = bounds[:, 1+2*n]
        yupper = bounds[:, 2+2*n]
        axes.plot(x, ylower, color=colors[n], ls=linestyles[n], lw=linewidths[n], label=label)
        axes.plot(x, yupper, color=colors[n], ls=linestyles[n], lw=linewidths[n])
        axes.fill_between(x, ylower, yupper, facecolor=colors[n], alpha=alpha)


def plot_confidence_of_mass(axes, bounds, colors=['red', 'blue', 'darkgreen'], linestyles=['-', '-', '-'], linewidths=[2.0, 2.0, 2.0], alpha=0.2, label=None):
    """
        Make a filled plot of the confidence regions from the bounds table.
        """
    
    # plot each confidence interval
    for n in range(len(colors)):
        m_low_high = np.array([[bounds[i, 0], bounds[i, 1+2*n], bounds[i, 2+2*n]] for i in range(len(bounds))
                               if (bounds[i, 1+2*n] > 0.0) and (bounds[i, 2+2*n] > 0.0)])
        axes.plot(m_low_high[:, 0], m_low_high[:, 1], color=colors[n], ls=linestyles[n], lw=linewidths[n], label=label)
        axes.plot(m_low_high[:, 0], m_low_high[:, 2], color=colors[n], ls=linestyles[n], lw=linewidths[n])
        axes.fill_between(m_low_high[:, 0], m_low_high[:, 1], m_low_high[:, 2], facecolor=colors[n], alpha=alpha)
