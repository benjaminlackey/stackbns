# This is a module of common functions used for binary systems.
# It also includes MCMC functions that are common to all EOSs,
# and don't belong in the mcmcutilities module that is meant to be generic and not
# specific to gravitational wave problems.


import numpy as np
import scipy.stats as stats
#import mcmcutilities as mu
import mcmcutilitiesnoplot as mu

# Define constants
C_CGS = 29979245800.0 # cm s^-1 (exact)
GTIMESMSUN_CGS = 1.32712440018e26; # cm^3 s^-2 very high precision
G_CGS = 6.67428e-8; # cm^3 g^-1 s^-2 CODATA-06 low precision
MSUN_CGS = GTIMESMSUN_CGS / G_CGS; # for consistency
MPC_CGS = 3.08568e24; # cm


################################################################################
############ Read Mathematica .m file to get Fisher matrix results #############
################################################################################


#def MathematicaLineToFloat(linestring):
#    """Convert a line from a Mathematica .m file to a list of floats.
#        
#        This function strips the characters [' ', ',', '}', '{', '\n'] from the beginning and end,
#        then replaces '*^' with 'e'.
#        """
#    
#    # strip the listed characters from the left and right (\n is a single character)
#    # replace the mathematica '*^' format with the standard 'e' format
#    # split at ', '. split can only handle one string.
#    stringlist = linestring.strip(' ,}{\n').replace('*^', 'e').split(', ')
#    return map(float, stringlist)
#
#def GetTrueBestCov(truebestcovstringlist):
#    """Return a true parameters list, best-fit parameters list, covariance matrix list.
#        """
#    
#    true = MathematicaLineToFloat(truebestcovstringlist[0])
#    best = MathematicaLineToFloat(truebestcovstringlist[1])
#    nParams = len(true)
#    cov = [MathematicaLineToFloat(truebestcovstringlist[i]) for i in range(2, nParams+2)]
#    
#    return true, best, cov
#
#def ReadTrueBestCovFile(filepath):
#    """Reads a Mathematica .m file containing a list of {true, best, covmat} values for each binary.
#        Returns 3 arrays: truelist, bestlist, covlist.
#        """
#    
#    # Open the file
#    f = open(filepath, 'r')
#    
#    # Read entire file as a single string
#    # Then split the string at each newline character '\n'
#    filelines = f.read().split('\n')
#    
#    nSystems = (len(filelines) - 2) / 5
#    
#    truelist = [0.0]*nSystems
#    bestlist = [0.0]*nSystems
#    covlist = [0.0]*nSystems
#    
#    # Read 5 lines at a time (skipping first line) until you get to the end of the file
#    for n in range(nSystems):
#        truebestcovstringlist = filelines[5*n+1:5*n+6]
#        truelist[n], bestlist[n], covlist[n] = GetTrueBestCov(truebestcovstringlist)
#    
#    f.close()
#    
#    return np.array(truelist), np.array(bestlist), np.array(covlist)


def ReadTrueCovFile(filepath):
    """
    Reads a Mathematica .m file containing a list of {true, covmat} values for each binary.
    
    Args:
        filepath (str): Path to Mathematica .m file
    Returns:
        truelist (2-d array): List of [ln(Mchirp), ln(eta), tildeLambda] values for each binary.
        covlist (3-d array): Corresponding list of covariance matrices for each binary.
    """
    
    # Open the file
    f = open(filepath, 'r')

    # Read entire file as a single string
    # Then split at first line enclosed by (* *)
    # Only use the second line, and strip the leading and trailing whitespace
    datastring = f.read().split('*)')[1]

    # Split datastring at all instances of whitespace
    # Join all the splitted strings together with the empty character
    datastring = "".join(datastring.split())

    # Split at end of each vector or row of matrix
    # then get rid of empty elements
    datalist = datastring.split('}')
    datalist = [value for value in datalist if value != '']

    # Strip other leading and trailing characters from line
    # then put Mathematica float into standard float form
    # then split line at each comma 
    stringmatrix = [string.strip('{,').replace('*^', 'e').split(',') for string in datalist]

    # Convert string to float
    matrix = [[float(stringmatrix[i][j]) for j in range(len(stringmatrix[0]))] for i in range(len(stringmatrix))]

    nParam = len(matrix[0])
    # There are 1+nParam lines per system (one for true parameters and nParam for the covariance matrix)
    nSystems = len(matrix) / (nParam + 1)

    truelist = [0.0]*nSystems
    covlist = [0.0]*nSystems

    # Number of lines per binary system
    shift = nParam + 1
    for n in range(nSystems):
        truelist[n] = matrix[shift*n]
        covlist[n] = matrix[shift*n + 1 : shift*n + shift]

    f.close()

    return np.array(truelist), np.array(covlist)


def ReadTrueBestCovFile(filepath):
    """Reads a Mathematica .m file containing a list of {true, best, covmat} values for each binary.
        Returns 3 arrays: truelist, bestlist, covlist.
        """
    
    # Open the file
    f = open(filepath, 'r')
    
    # Read entire file as a single string
    # Then split at first line enclosed by (* *)
    # Only use the second line, and strip the leading and trailing whitespace
    datastring = f.read().split('*)')[1]
    
    # Split datastring at all instances of whitespace
    # Join all the splitted strings together with the empty character
    datastring = "".join(datastring.split())
    
    # Split at end of each vector or row of matrix
    # then get rid of empty elements
    datalist = datastring.split('}')
    datalist = [value for value in datalist if value != '']
    
    # Strip other leading and trailing characters from line
    # then put Mathematica float into standard float form
    # then split line at each comma 
    stringmatrix = [string.strip('{,').replace('*^', 'e').split(',') for string in datalist]
    
    # Convert string to float
    matrix = [[float(stringmatrix[i][j]) for j in range(len(stringmatrix[0]))] for i in range(len(stringmatrix))]
    
    nParam = len(matrix[0])
    nSystems = len(matrix) / (nParam + 2)
    
    truelist = [0.0]*nSystems
    bestlist = [0.0]*nSystems
    covlist = [0.0]*nSystems

    # Number of lines per binary system
    shift = nParam + 2
    for n in range(nSystems):
        truelist[n] = matrix[shift*n]
        bestlist[n] = matrix[shift*n + 1]
        covlist[n] = matrix[shift*n + 2 : shift*n + shift]
    
    f.close()
    
    return np.array(truelist), np.array(bestlist), np.array(covlist)


################################################################################
#     Functions for extracting information from LALInference MCMC results.     #
################################################################################


def ReadLALMCMCRun(filepath, paramnames):
    """Read in output from MCMC run.
        Outputs a chain for the parameters paramnames.
        """
    
    f = open(filepath, 'r')
    labelstring = f.readline()
    labelnames = labelstring.split() # split without arguments splits on any whitespace
    
    # Get column number for each name in paramnames
    # enumerate gives list of tuples [(0, element_0), ...]
    columns = [0]*len(paramnames)
    for i,name in enumerate(paramnames):
        for j,label in enumerate(labelnames):
            if name == label:
                columns[i] = j
    
    # get the chain (only containing parameters listed in paramnames)
    chain = []
    for line in f:
        # Split line at whitespace then convert string to float
        linkstring = map(float, line.split())
        link = map(float, linkstring)
        chain.append(np.array(link)[columns])

    f.close()

    return np.array(chain)


def MaxCoordinatesFromHistogramND(histnd, centers):
    """ Get indices corresponding to maximum value in histnd.
        histnd is found from CenteredHistogramND in mcmcutilities.
        Used for finding maximum posterior from MCMC run.
        """
    maxindices = np.unravel_index(histnd.argmax(), histnd.shape)
    
    # Get coordinates of maximum bin
    maximum = [centers[i, maxindices[i]] for i in range(len(maxindices))]
    return np.array(maximum)


def SetMarginalizedEventLikeFromMCMCRun(filepath):
    """ Get MCMC run file, and extract Mchirp, q, lambdatilde.
        Then use the Gaussian kernel density estimator to generate p(lnMchirp, lnEta, tLambda).
        """
    
    paramnames = ['mc', 'q', 'lambdat']
    chain = ReadLALMCMCRun(filepath, paramnames)
    
    chain[:, 0] = np.log(chain[:, 0]) # convert MChirp to lnMChirp
    chain[:, 1] = np.log(EtaOfQ(chain[:, 1])) # convert q to lnEta
    
    return stats.gaussian_kde(chain.T)


def GetApproximateMaxLike(filepath):
    """Get MCMC run file, and extract (Mchirp, q, lambdatilde).
        Then convert to (lnMchirp, lnEta, tLambda).
        Histogram and smooth then find largest bin point.
        """
    
    paramnames = ['mc', 'q', 'lambdat']
    chain = ReadLALMCMCRun(filepath, paramnames)
    
    chain[:, 0] = np.log(chain[:, 0]) # convert MChirp to lnMChirp
    chain[:, 1] = np.log(EtaOfQ(chain[:, 1])) # convert q to lnEta
    
    histnd, centers = mu.CenteredHistogramND(chain, bins=30, histrange=None, smooth=True)
    
    return MaxCoordinatesFromHistogramND(histnd, centers)


################################################################################
##################  Common functions for binary systems       ##################
################################################################################


# Mchirp and eta do not depend on which mass (m1 or m2) is greater.
# Going backwards requires a choice for which mass is greater.

def MChirpOfM1M2(m1, m2):
    return (m1*m2)**(3.0/5.0) / (m1+m2)**(1.0/5.0)

def EtaOfM1M2(m1, m2):
    return (m1*m2) / (m1+m2)**2.0

def EtaOfQ(q):
    """Takes either big Q=m_1/m_2 or little q=m_2/m_1 and returns
        symmetric mass ratio eta.
        """
    return q / (1.0 + q)**2

# M1 is always the more massive star (the primary)
def M1OfMChirpEta(mChirp, eta):
    return (1.0/2.0)*mChirp*eta**(-3.0/5.0) * (1.0 + np.sqrt(1.0-4.0*eta))

# M2 is always the less massive star (the secondary)
def M2OfMChirpEta(mChirp, eta):
    return (1.0/2.0)*mChirp*eta**(-3.0/5.0) * (1.0 - np.sqrt(1.0-4.0*eta))

def LambdaTildeOfEtaL1L2(eta, lambda1, lambda2):
    """$\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$. 
        Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
        Lambda_2 is for the secondary star m_2.
        """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lambda1+lambda2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lambda1-lambda2))

# This is the definition found in Les Wade's paper.
# Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper (I think).
# Ask Les to please change this.
def DeltaLambdaTildeOfEtaL1L2(eta, lambda1, lambda2):
    """$\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$. 
        Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
        Lambda_2 is for the secondary star m_2.
        """
    return (1.0/2.0)*(
                      np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(lambda1+lambda2)
                      + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(lambda1-lambda2)
                      )


################################################################################
################## Define mass prior and Gaussian likelihood  ##################
################################################################################

def LogMassPrior(lnMChirp, lnEta):
    minMass, maxMass = 0.5, 3.0
    
    if np.exp(lnEta) > 0.25:
        return mu.logZero
    elif M2OfMChirpEta(np.exp(lnMChirp), np.exp(lnEta)) < minMass:
        return mu.logZero
    elif M1OfMChirpEta(np.exp(lnMChirp), np.exp(lnEta)) > maxMass:
        return mu.logZero
    else:
        return 0.0

# The likelihood function for each event is a multivariate Gaussian determined by the Fisher matrix
# It is proportional to the ln of the Gaussian (doesn't have normalizing factor in front)
# takes the inverse covariane matrix invCov and the bestFit for the 3 parameters
def LogMarginalizedEventLike(lnMChirp, lnEta, tLambda, bestFit, invCov):
    x = np.array([lnMChirp, lnEta, tLambda])
    diff = x - bestFit
    return -np.dot(diff,np.dot(invCov,diff))/2.0 # -(1/2) (x-mu)^T Sigma^{-1} (x-mu)

