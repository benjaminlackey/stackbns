#idea: make a class called EOS which contains methods for generating MMax(parameters)

import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage # used for interpolation
import emcee
import binaryutilities as binary
#import mcmcutilities as mu
import mcmcutilitiesnoplot as mu
#matplotlib isn't actually used in this file
#import matplotlib.pyplot as plt

################################################################################
##################         Functions for generating EOS       ##################
################################################################################


def Set4ParameterPiecewisePolytrope(logp1, gamma1, gamma2, gamma3):
    
    # 4-piece low-density fit to SLy4
    # gives pressure in dyn/cm^2
    # rest-mass density in g/cm^3 at START of polytropic piece
    rhoLow = [0.0, 2.44033979e7, 3.78358138e11, 2.62780487e12]
    kLow = [6.11252036792443e12, 9.54352947022931e14, 4.787640050002652e22, 3.593885515256112e13]
    gammaLow = [1.58424999, 1.28732904, 0.62223344, 1.35692395]
    
    rho1 = 10**14.7
    rho2 = 10**15.0
    
    # pressure at rho1CGS
    p1 = 10**logp1
    
    # pressure constants
    k1 = p1/rho1**gamma1
    k2 = p1/rho1**gamma2
    k3 = k2*rho2**(gamma2 - gamma3)
    
    # calculate the variable joining density rho0 between the high and low density EOS
    rho0 = (kLow[3]/k1)**(1.0/(gamma1-gammaLow[3]))
    
    # Add another polytrope if the joining density is below the start of the last low density polytrope or
    # above the end of the first high density polytrope.
    
    if (rho0 > rhoLow[3]) and (rho0 < rho1):
        # No issue. There will be a total of 7 polytropes. *)
        kTab = kLow + [k1, k2, k3]
        gammaTab = gammaLow + [gamma1, gamma2, gamma3]
        rhoTab = rhoLow + [rho0, rho1, rho2]
    else:
        # You have to add an 8th polytrope between gammaLowCGS[3] and gamma1
        # It will be between the densities rhoJoin1 and rhoJoin2
        rhoJoin1 = 5.0e12
        rhoJoin2 = 1.0e13
        # Calculate the pressure at the start and end densities
        pJoin1 = kLow[3]*rhoJoin1**gammaLow[3]
        pJoin2 = k1*rhoJoin2**gamma1
        # Calculate k and gamma for the joining polytrope
        gammaJoin = log(pJoin2/pJoin1) / log(rhoJoin2/rhoJoin1)
        kJoin = pJoin1/rhoJoin1**gammaJoin
        # Now join all 8 polytropes
        kTab = kLow + [kJoin, k1, k2, k3]
        gammaTab = gammaLow + [gammaJoin, gamma1, gamma2, gamma3]
        rhoTab = rhoLow + [rhoJoin1, rhoJoin2, rho1, rho2]
        print("An extra polytrope was used to join the low and high density regions.")
    
    return kTab, gammaTab, rhoTab


def POfRhoPiecewisePolytrope(rho, kTab, gammaTab, rhoTab):
    
    # get index for polytrope piece
    i = len(rhoTab) - 1
    while (rho <= rhoTab[i]) and (i > 0):
        i -= 1
    
    return kTab[i] * rho ** gammaTab[i]


############### Interpolate MMax(EOS), VMax(EOS), R(EOS, M), k_2(EOS, M), Lambda(EOS, M) #############

def InterpolateUniformArray(data, xmin, xmax, x):
    """Interpolate N-d function from uniformly spaced N-d array.
        xmin, xmax, x are arrays of length N.
        """
    
    # get number of points in each dimension
    Nx = np.array(data.shape)
    
    # rescale to pixel coordinates (0 to N-1 in each dimension)
    xRescale = (x-xmin) * (Nx-1.0) / (xmax-xmin)
    
    # list of [[x], [y], [z], ...] coordinates for one point
    xList = np.array([xRescale]).T.tolist()
    
    # interpolate using the function that assumes pixel coordinates
    return ndimage.map_coordinates(data, xList, order=1)[0]


constraintsArray = np.loadtxt('/home/blackey/MeasureEOS/data/4ParamEOSConstraints.dat')

lpmin, lpmax = constraintsArray[0, 0], constraintsArray[-1, 0]
g1min, g1max = constraintsArray[0, 1], constraintsArray[-1, 1]
g2min, g2max = constraintsArray[0, 2], constraintsArray[-1, 2]
g3min, g3max = constraintsArray[0, 3], constraintsArray[-1, 3]
mmin, mmax = constraintsArray[0, 6], constraintsArray[0, -3]



# reshape array to 4-d with lp varying slowest and gamma3 varying fastest
# !!!!!!!!!!!!! figure out how to get number of points in each direction automatically !!!!!!!!!!!!!!
mMaxArray = constraintsArray[:, 4].reshape(16, 16, 16, 16)
vMaxArray = constraintsArray[:, 5].reshape(16, 16, 16, 16)

# min and max values of coordinates for interpolation
p123min = np.array([lpmin, g1min, g2min, g3min])
p123max = np.array([lpmax, g1max, g2max, g3max])

def MMaxOfP123(lp, g1, g2, g3):
    p123 = np.array([lp, g1, g2, g3])
    return InterpolateUniformArray(mMaxArray, p123min, p123max, p123)

def VMaxOfP123(lp, g1, g2, g3):
    p123 = np.array([lp, g1, g2, g3])
    return InterpolateUniformArray(vMaxArray, p123min, p123max, p123)

#mMaxOfP123InterpolateObject = interpolate.LinearNDInterpolator(constraintsArray[:, [0, 1, 2, 3]], constraintsArray[:, 4])
#def MMaxOfP123(lp, g1, g2, g3):
#    return mMaxOfP123InterpolateObject([[lp, g1, g2, g3]])[0]
#
#vMaxOfP123InterpolateObject = interpolate.LinearNDInterpolator(constraintsArray[:, [0, 1, 2, 3]], constraintsArray[:, 5])
#def VMaxOfP123(lp, g1, g2, g3):
#    return vMaxOfP123InterpolateObject([[lp, g1, g2, g3]])[0]

# Put data into proper format to get interpolated functions R(lp, g1, g2, g3, m), k2(lp, g1, g2, g3, m), Lambda(lp, g1, g2, g3, m)
lenarray = len(constraintsArray)
ncol = len(constraintsArray[0])
masses = np.arange(6, ncol, 3)
nmasses = len(masses)
p123mrkArray = np.zeros((lenarray*nmasses, 7))

for i in range(lenarray):
    for j in range(nmasses):
        p123mrkArray[i*nmasses+j] = np.array([constraintsArray[i, 0], constraintsArray[i, 1], constraintsArray[i, 2], constraintsArray[i, 3], constraintsArray[i, 6+3*j], constraintsArray[i, 7+3*j], constraintsArray[i, 8+3*j]])

# reshape array to 5-d with lp varying slowest and m varying fastest
# !!!!!!!!!!!!! figure out how to get number of points in each direction automatically !!!!!!!!!!!!!!
radiusArray = p123mrkArray[:, 5].reshape(16, 16, 16, 16, nmasses)
k2Array = p123mrkArray[:, 6].reshape(16, 16, 16, 16, nmasses)

# min and max values of coordinates for interpolation
p123mmin = np.array([lpmin, g1min, g2min, g3min, mmin])
p123mmax = np.array([lpmax, g1max, g2max, g3max, mmax])


#ROfP123MInterpolateObject = interpolate.LinearNDInterpolator(p123mrkArray[:, [0, 1, 2, 3, 4]], p123mrkArray[:, 5])
#def ROfP123M(lp, g1, g2, g3, mass):
#    return ROfP123MInterpolateObject([[lp, g1, g2, g3, mass]])[0]
#
#K2OfP123MInterpolateObject = interpolate.LinearNDInterpolator(p123mrkArray[:, [0, 1, 2, 3, 4]], p123mrkArray[:, 6])
#def K2OfP123M(lp, g1, g2, g3, mass):
#    return K2OfP123MInterpolateObject([[lp, g1, g2, g3, mass]])[0]

def ROfP123M(lp, g1, g2, g3, mass):
    p123m = np.array([lp, g1, g2, g3, mass])
    return InterpolateUniformArray(radiusArray, p123mmin, p123mmax, p123m)

def K2OfP123M(lp, g1, g2, g3, mass):
    p123m = np.array([lp, g1, g2, g3, mass])
    return InterpolateUniformArray(k2Array, p123mmin, p123mmax, p123m)

def LambdaOfP123M(lp, g1, g2, g3, mass):
    return (2.0/3.0) * K2OfP123M(lp, g1, g2, g3, mass) * (binary.C_CGS**2*ROfP123M(lp, g1, g2, g3, mass)*10.0**5 / (binary.G_CGS*binary.MSUN_CGS*mass))**5

######################### Fit for SLY4 low-density EOS #########################
# Gives pressure in dyne/cm^2

kLowCGS = np.array([6.11252036792443e12, 9.54352947022931e14, 4.787640050002652e22, 3.593885515256112e13])
gammaLowCGS = np.array([1.58424999, 1.28732904, 0.62223344, 1.35692395])
rhoLowCGS = np.array([0.0, 2.44033979e7, 3.78358138e11, 2.62780487e12])

rho1CGS = 10**14.7;
rho2CGS = 10**15.0;

pLowCGS = kLowCGS * rhoLowCGS**gammaLowCGS

def P1UpperBound(g1):
    return pLowCGS[3]*(rho1CGS/rhoLowCGS[3])**g1

def LogP1UpperBound(g1):
    return np.log10(P1UpperBound(g1))


################################################################################
#       Define priors (same for both Fisher matrix and LALInferenceMCMC)       #
################################################################################


def LogEOSPrior(lp, g1, g2, g3, massKnown=1.93, causalLimit=1.0):
    lpmin, lpmax = 33.5, 35.5
    g1min, g1max = 1.4, 5.0
    g2min, g2max = 1.08, 5.0
    g3min, g3max = 1.08, 5.0
    # Ordered from fastest to slowest to compute
    if lp<lpmin or lp>lpmax or g1<g1min or g1>g1max or g2<g2min or g2>g2max or g3<g3min or g3>g3max:
        return mu.logZero
    elif lp > LogP1UpperBound(g1):
        return mu.logZero
    elif MMaxOfP123(lp, g1, g2, g3) < massKnown:
        return mu.logZero
    elif VMaxOfP123(lp, g1, g2, g3) > causalLimit:
        return mu.logZero
    else:
        return 0.0


# The complete prior
def LogPrior(params, nSystems, massKnown=1.93, causalLimit=1.0):
    logMassPriorList = [0.0]*nSystems
    for n in range(len(logMassPriorList)):
        logMassPriorList[n] = binary.LogMassPrior(params[4+2*n], params[5+2*n])
    
    return LogEOSPrior(params[0], params[1], params[2], params[3], massKnown=massKnown, causalLimit=causalLimit) + sum(logMassPriorList)


def LambdaTildeOfParams(lp, g1, g2, g3, mChirp, eta):
    """$\tilde\Lambda(\log(p_1), \Gamma, \mathcal{M}, \eta)$
        """
    # m1 > m2
    m1 = binary.M1OfMChirpEta(mChirp, eta)
    m2 = binary.M2OfMChirpEta(mChirp, eta)
    
    lambda1 = LambdaOfP123M(lp, g1, g2, g3, m1)
    lambda2 = LambdaOfP123M(lp, g1, g2, g3, m2)
    
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lambda1+lambda2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lambda1-lambda2))


################################################################################
#  Functions for generating initial points for walkers and running emcee       #
################################################################################


def SampleSingleInitialWalkerPoint(centerArray, widthArray, massKnown=1.93, causalLimit=1.0):
    """The initial point for a single walker at a single temperature.
        Draw from a uniform distribution centered on centerList with width widthList.
        Reject point if it is outside the prior.
        """
    
    minArray = centerArray - 0.5*widthArray
    maxArray = centerArray + 0.5*widthArray
    
    ndim = len(centerArray)
    nSystems = (ndim-4)/2 # !!! This depends on the number of EOS parameters !!!
    
    # Initialize parameter array for the single walker initial point p0
    p0 = np.array([0.0]*ndim)
    
    # Sample EOS parameters
    # the equivalent of a do while loop:
    while True:
        #draw point
        lp = np.random.uniform(minArray[0], maxArray[0])
        g1 = np.random.uniform(minArray[1], maxArray[1])
        g2 = np.random.uniform(minArray[2], maxArray[2])
        g3 = np.random.uniform(minArray[3], maxArray[3])
        p0[[0, 1, 2, 3]] = [lp, g1, g2, g3]
        if LogEOSPrior(lp, g1, g2, g3, massKnown=massKnown, causalLimit=causalLimit) != mu.logZero:
            break
    
    
    # Sample parameters for each system
    for n in range(nSystems):
        while True:
            #draw point
            lnMChirp = np.random.uniform(minArray[2*n+4], maxArray[2*n+4])
            lnEta = np.random.uniform(minArray[2*n+5], maxArray[2*n+5])
            p0[[2*n+4, 2*n+5]] = [lnMChirp, lnEta]
            if binary.LogMassPrior(lnMChirp, lnEta) != mu.logZero:
                break
    
    return p0


def SampleInitialWalkerPoints(nwalkers, centerArray, widthArray, massKnown=1.93, causalLimit=1.0):
    """The initial points for the walkers at each temperature.
        Draw from a uniform distribution centered on centerList with width widthList.
        Reject point if it is outside the prior.
        """
    
    assert (len(centerArray) == len(widthArray)),"Length of arguments 2 and 3 are not the same."
    
    ndim = len(centerArray)
    points0 = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        points0[i] = SampleSingleInitialWalkerPoint(centerArray, widthArray, massKnown=massKnown, causalLimit=causalLimit)
    
    return points0


######                 Using Fisher matrix approximation                   #####


################################################################################
#       Define Likelihoods, and posteriors from Fisher matrix results.         #
################################################################################


def LogEventLike(lp, g1, g2, g3, lnMChirp, lnEta, bestFit, invCov):
    # LambdaTildeOfParams may be undefined when not in prior regions,
    # so just return logZero if outside the prior.
    # An MCMC point outside the prior will never be accepted anyway.
    if binary.LogMassPrior(lnMChirp, lnEta) == mu.logZero:
        return mu.logZero
    elif LogEOSPrior(lp, g1, g2, g3) == mu.logZero:
        return mu.logZero
    else:
        tLambda = LambdaTildeOfParams(lp, g1, g2, g3, np.exp(lnMChirp), np.exp(lnEta))
        return binary.LogMarginalizedEventLike(lnMChirp, lnEta, tLambda, bestFit, invCov)


def LogLike(params, nSystems, bestFitArray, invCovArray):
    logEventLikeArray = [0.0]*nSystems
    for n in range(len(logEventLikeArray)):
        logEventLikeArray[n] = LogEventLike(params[0], params[1], params[2], params[3], params[4+2*n], params[5+2*n], bestFitArray[n], invCovArray[n])
    
    return sum(logEventLikeArray)


def LogPost(params, nSystems, bestFitArray, invCovArray, massKnown=1.93, causalLimit=1.0):
    return LogPrior(params, nSystems, massKnown=massKnown, causalLimit=causalLimit) + LogLike(params, nSystems, bestFitArray, invCovArray)


########################################################################################################
#                             Running emcee using Fisher matrix results                             #
########################################################################################################

def RunMCMCFor4ParamEOSZeroNoise(trueArray, covArray, nWalkers, nBurn, nSteps, threads=1, massKnown=1.93, causalLimit=1.0):
    # Take the inverse for each cov matrix in the array covArray
    invCovArray = np.array(map(np.linalg.inv, covArray))

    # Dimension of the parameter space (EOS parameters + 2 mass parameters for each system)
    nSystems = len(trueArray)
    nDim = 4+2*nSystems

    # generate array of starting points for the nwalkers
    # Center on the true values (the bestFit values may be outside the prior)
    centerEOS = np.array([34.7, 2.7, 2.7, 2.7])
    centerMasses = np.reshape(trueArray[:, [0, 1]], 2*nSystems)
    centerArray = np.concatenate([centerEOS, centerMasses])
    widthArray = np.concatenate([[1.0, 5.0, 5.0, 5.0], [0.001, 0.01]*nSystems])
    pointsWalkers = SampleInitialWalkerPoints(nWalkers, centerArray, widthArray, massKnown=massKnown, causalLimit=causalLimit)

    # Initialize the sampler.
    # the arguments args are extra arguments to lnprob
    sampler = emcee.EnsembleSampler(nWalkers, nDim, LogPost, args=[nSystems, trueArray, invCovArray, massKnown, causalLimit], threads=threads)

    # Run nBurn steps as a burn-in.
    # pointsWalkers is the current position of the walkers
    # lnPostWalkers is the current ln(posterior) for each of the walkers
    # state stores the current state for the EnsembleSampler named sampler. Used for restarting the run.
    pointsWalkers, lnPostWalkers, state = sampler.run_mcmc(pointsWalkers, nBurn)
    print("Acceptance fraction for each walker during burn-in:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during burn-in:\n", sampler.acor)

    # Reset the chain to remove the burn-in samples.
    sampler.reset()

    # Starting from the final position in the burn-in chain, sample for nStep steps.
    # rstate0: initial state to use for the random number generator
    sampler.run_mcmc(pointsWalkers, nSteps, rstate0=state) # returns pos, prob, state after nStep steps
    print("Acceptance fraction for each walker during run:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during run:\n", sampler.acor)

    #chain = sampler.flatchain
    # first index is walker number. second index is step of chain for each walker. third is parameter number.
    #chain = sampler.chain
    #
    ## transpose walker number and chain step, then flatten.
    #return reshape(np.transpose(chain, (1, 0, 2)), (nWalkers*nSteps, nDim))

    return sampler.chain


def RunMCMCFor4ParamEOS(trueArray, bestFitArray, covArray, nWalkers, nBurn, nSteps, threads=1, massKnown=1.93, causalLimit=1.0):
    # Take the inverse for each cov matrix in the array covArray
    invCovArray = np.array(map(np.linalg.inv, covArray))
    
    # Dimension of the parameter space (EOS parameters + 2 mass parameters for each system)
    nSystems = len(bestFitArray)
    nDim = 4+2*nSystems
    
    # generate array of starting points for the nwalkers
    # Center on the true values (the bestFit values may be outside the prior)
    centerEOS = np.array([34.7, 2.7, 2.7, 2.7])
    centerMasses = np.reshape(trueArray[:, [0, 1]], 2*nSystems)
    centerArray = np.concatenate([centerEOS, centerMasses])
    widthArray = np.concatenate([[1.0, 5.0, 5.0, 5.0], [0.001, 0.01]*nSystems])
    pointsWalkers = SampleInitialWalkerPoints(nWalkers, centerArray, widthArray, massKnown=massKnown, causalLimit=causalLimit)
    
    # Initialize the sampler.
    # the arguments args are extra arguments to lnprob
    sampler = emcee.EnsembleSampler(nWalkers, nDim, LogPost, args=[nSystems, bestFitArray, invCovArray, massKnown, causalLimit], threads=threads)
    
    # Run nBurn steps as a burn-in.
    # pointsWalkers is the current position of the walkers
    # lnPostWalkers is the current ln(posterior) for each of the walkers
    # state stores the current state for the EnsembleSampler named sampler. Used for restarting the run.
    pointsWalkers, lnPostWalkers, state = sampler.run_mcmc(pointsWalkers, nBurn)
    print("Acceptance fraction for each walker during burn-in:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during burn-in:\n", sampler.acor)
    
    # Reset the chain to remove the burn-in samples.
    sampler.reset()
    
    # Starting from the final position in the burn-in chain, sample for nStep steps.
    # rstate0: initial state to use for the random number generator
    sampler.run_mcmc(pointsWalkers, nSteps, rstate0=state) # returns pos, prob, state after nStep steps
    print("Acceptance fraction for each walker during run:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during run:\n", sampler.acor)
    
    #chain = sampler.flatchain
    # first index is walker number. second index is step of chain for each walker. third is parameter number.
    #chain = sampler.chain
    #
    ## transpose walker number and chain step, then flatten.
    #return reshape(np.transpose(chain, (1, 0, 2)), (nWalkers*nSteps, nDim))
    
    return sampler.chain


######                       Using LALInferenceMCMC                        #####


################################################################################
#  Define Likelihoods, and posteriors from LALInferenceMCMC results.   #
################################################################################


#def LogEventLike(lp, g1, g2, g3, lnMChirp, lnEta, bestFit, invCov):
#    # LambdaTildeOfParams may be undefined when not in prior regions,
#    # so just return logZero if outside the prior.
#    # An MCMC point outside the prior will never be accepted anyway.
#    if binary.LogMassPrior(lnMChirp, lnEta) == mu.logZero:
#        return mu.logZero
#    elif LogEOSPrior(lp, g1, g2, g3) == mu.logZero:
#        return mu.logZero
#    else:
#        tLambda = LambdaTildeOfParams(lp, g1, g2, g3, np.exp(lnMChirp), np.exp(lnEta))
#        return binary.LogMarginalizedEventLike(lnMChirp, lnEta, tLambda, bestFit, invCov)


def LogEventLikeFromMCMC(lp, g1, g2, g3, lnMChirp, lnEta, MarginalizedEventLike):
    """Take MarginalizedEventLike(lnMChirp, lnEta, tLambda) function determined by KDE of MCMC run.
        Return the Log of the event likelihood determined by (lp, gamma, lnMChirp, lnEta).
        """
    
    # LambdaTildeOfParams may be undefined when not in prior regions,
    # so just return logZero if outside the prior.
    # An MCMC point outside the prior will never be accepted anyway.
    if binary.LogMassPrior(lnMChirp, lnEta) == mu.logZero:
        return mu.logZero
    elif LogEOSPrior(lp, g1, g2, g3) == mu.logZero:
        return mu.logZero
    else:
        tLambda = LambdaTildeOfParams(lp, g1, g2, g3, np.exp(lnMChirp), np.exp(lnEta))
        # MarginalizedEventLike is a function generated from binary.SetMarginalizedEventLikeFromMCMCRun
        like = MarginalizedEventLike([lnMChirp, lnEta, tLambda])[0]
        # like will evaluate to 0.0 far from max
        if like == 0.0:
            return mu.logZero
        else:
            return np.log(like)


def LogLikeFromMCMC(params, MarginalizedEventLikeArray):
    """
        Multiply the likelihood functions for each event together.
        There is probably a more efficient Python way to do this summation.
        """
    nSystems = len(MarginalizedEventLikeArray)
    #print(nSystems)
    logEventLikeArray = [0.0]*nSystems
    for n in range(nSystems):
        logEventLikeArray[n] = LogEventLikeFromMCMC(params[0], params[1], params[2], params[3], params[4+2*n], params[5+2*n], MarginalizedEventLikeArray[n])
    
    #print(logEventLikeArray)
    return sum(logEventLikeArray)


def LogPostFromMCMC(params, MarginalizedEventLikeArray, massKnown=1.93, causalLimit=1.0):
    nSystems = len(MarginalizedEventLikeArray)
    return LogPrior(params, nSystems, massKnown=massKnown, causalLimit=causalLimit) + LogLikeFromMCMC(params, MarginalizedEventLikeArray)


########################################################################################################
#                             Running emcee using LALInferenceMCMC results                             #
########################################################################################################


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This only seems to work if you set threads=1. See emcee webpage.
# Something about statistics functions (I'm guessing the kde) must be pickleable.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def RunMCMCFor4ParamEOSWithLikeFromMCMC(filepathlist, nWalkers, nBurn, nSteps, massKnown=1.93, causalLimit=1.0):
    """filepathlist is list of LALInferenceMCMC runs.
        """
    # Get functions for eventlikelihoods from the MCMC files
    MarginalizedEventLikeList = np.array([binary.SetMarginalizedEventLikeFromMCMCRun(fp) for fp in filepathlist])
    
    # Dimension of the parameter space (EOS parameters + 2 mass parameters for each system)
    nSystems = len(filepathlist)
    nDim = 4+2*nSystems
    
    # Get maximum likelihood for starting values for MCMC chains
    maxArray = np.array([binary.GetApproximateMaxLike(fp) for fp in filepathlist])
    
    # generate array of starting points for the nwalkers
    centerEOS = np.array([34.7, 2.7, 2.7, 2.7])
    centerMasses = np.reshape(maxArray[:, [0, 1]], 2*nSystems)
    centerArray = np.concatenate([centerEOS, centerMasses])
    widthArray = np.concatenate([[1.0, 5.0, 5.0, 5.0], [0.0001, 0.001]*nSystems])
    pointsWalkers = SampleInitialWalkerPoints(nWalkers, centerArray, widthArray, massKnown=massKnown, causalLimit=causalLimit)
        
    # Initialize the sampler.
    # the arguments args are extra arguments to lnprob
    sampler = emcee.EnsembleSampler(nWalkers, nDim, LogPostFromMCMC, args=[MarginalizedEventLikeList, massKnown, causalLimit], threads=1)
    
    # Run nBurn steps as a burn-in.
    # pointsWalkers is the current position of the walkers
    # lnPostWalkers is the current ln(posterior) for each of the walkers
    # state stores the current state for the EnsembleSampler named sampler. Used for restarting the run.
    pointsWalkers, lnPostWalkers, state = sampler.run_mcmc(pointsWalkers, nBurn)
    print("Acceptance fraction for each walker during burn-in:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during burn-in:\n", sampler.acor)
    
    # Reset the chain to remove the burn-in samples.
    sampler.reset()
    
    # Starting from the final position in the burn-in chain, sample for nStep steps.
    # rstate0: initial state to use for the random number generator
    sampler.run_mcmc(pointsWalkers, nSteps, rstate0=state) # returns pos, prob, state after nStep steps
    print("Acceptance fraction for each walker during run:\n", sampler.acceptance_fraction)
    print("Autocorrelation time for each parameter during run:\n", sampler.acor)
    
    return sampler.chain

