"""
python postprocessfiteos.py --file_prefix_list \
mcmc4pbns1snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
mcmc4pbns5snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
mcmc4pbns10snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
mcmc4pbns15snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
mcmc4pbns20snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
mcmc4pbns50snr8m12to16fISCOLALMCMCmpa1fitTaylorF2. \
--file_run_list 0 1 2 3 4 5 6 7 8 9 \
--file_suffix .npy \
--nsystems_list 1 5 10 15 20 50 \
--eos mpa1 \
--eos_type tab
--output test.pkl
"""

import argparse
import numpy as np
import pickle
import binaryutilities as binary

############## constants ###################

sigma1 = 0.68268949 # ~317 points out of 1000 are outside
sigma2 = 0.95449974 # ~46 points out of 1000 are outside
sigma3 = 0.99730024 # ~3 points out of 1000 are outside

conf = [sigma1, sigma2, sigma3]

massmin = 0.05
massmax = 3.2
nmass = 50
#nmass = 2
#__name__ is the name of the current file
#__main__ is the name of the program that was called from the shell
if __name__ == "__main__":
    
    # Create ArgumentParser object named parser
    parser = argparse.ArgumentParser(description="Calculate confidence intervals for (R(M), lambda(M)).")
    
    # Add positional arguments to parser object
    # Files for input data
    # "+" makes a list of the arguments
    parser.add_argument("--file_prefix_list", nargs="+", required=True, help="List of the prefix input files from the emcee runs.")
    parser.add_argument("--file_run_list", nargs="+", required=True, help="List of strings identifying the individual runs.")
    parser.add_argument("--file_suffix", required=True, help="File name after the number string")
    parser.add_argument("--nsystems_list", nargs="+", type=int, required=True, help="List of the number of BNS systems in each run.")
    # Name of output file
    parser.add_argument("--output", required=True, help="Output pickle (.pkl) file.")
    
    # Do the argument parsing
    args = parser.parse_args()
    
    file_prefix_list = args.file_prefix_list
    file_run_list = args.file_run_list
    file_suffix = args.file_suffix
    output_file = args.output
    
    # The eos4paramconfidence module is time consuming, so import it after you have parsed the arguments.
    import eos4parameterpolytrope as eos4p
    import eos4paramconfidenceonemode as confidence
    
    # Get list of joined MCMC chains
    eoschainlist = [confidence.load_multiple_emcee_runs(file_prefix, file_run_list, file_suffix, nBurn=50, dSteps=2, flatten=True) for file_prefix in file_prefix_list]
    
    ##### Calculate the confidence contours (R(M), lambda(M)) #####
    
    # Calculate radius (ignoring BH samples)
    print('Starting one mode radius confidence bound calculations')
    quantity = 'radius'
    radiusmodeboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf)
    	print(bounds)
        radiusmodeboundslist.append(bounds)


    # Calculate lambda (ignoring BH samples)
    print('Starting one mode lambda confidence bound calculations')
    quantity = 'lambda'
    lambdamodeboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf)
    	print(bounds)
        lambdamodeboundslist.append(bounds)


    ##################### combine and save data ###################

    outdict = {'radiusmode':radiusmodeboundslist, 'lambdamode':lambdamodeboundslist}
    #outdict = {'nsystems':nsystems_list, 'logp':logpboundslist, 'perror':perrorboundslist, 'radius':radiusboundslist, 'lambda':lambdaboundslist}
    print(outdict)
     
    # Pickle data
    file = open(output_file, 'w')
    pickle.dump(outdict, file)
    file.close()
