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

rhomin = 10.0**14.0
rhomax = 10.0**15.5
nrho = 50
#nrho = 2

massmin = 0.05
massmax = 3.2
nmass = 50
#nmass = 2

eosfitdict = {
'sly': [34.384, 3.005, 2.988, 2.851], # SLy R14 = 11.736km
'eng': [34.437, 3.514, 3.130, 3.168], # ENG R14 = 12.059km
'mpa1':[34.495, 3.446, 3.572, 2.887], # MPA1 R14 = 12.473km
'ms1': [34.858, 3.224, 3.033, 1.325], # MS1 R14 = 14.918km
'ms1b':[34.855, 3.456, 3.011, 1.425], # MS1b R14 = 14.583km
'h4':  [34.669, 2.909, 2.246, 2.144], # H4 R14 = 13.774km
'alf2':[34.616, 4.070, 2.411, 1.890] # ALF2 R14 = 13.188km
}

# File names for rest-mass density rho, pressure p, energy density epsilon in g/cm^3.
rpefilelist = [
'/home/blackey/MeasureEOS/data/sly4.dat',
'/home/blackey/MeasureEOS/data/engvik.dat',
'/home/blackey/MeasureEOS/data/mpa1.dat',
'/home/blackey/MeasureEOS/data/ms00.dat',
'/home/blackey/MeasureEOS/data/ms2.dat',
'/home/blackey/MeasureEOS/data/H4.dat',
'/home/blackey/MeasureEOS/data/Alf_rho3.0_c0.3_mixed.dat'
]

#Load file for rest-mass density rho, pressure p, energy density epsilon in g/cm^3.
#Return rho (in g/cm^3) and p (in dyne/cm^2).
rparraylist = []    
for rpefile in rpefilelist:
    rpearray = np.loadtxt(rpefile)
    rarray = rpearray[:, 0]
    parray = rpearray[:, 1]*binary.C_CGS**2    
    rparraylist.append(np.transpose(np.array([rarray, parray])))

eostabdict = dict(zip(['sly', 'eng', 'mpa1', 'ms1', 'ms1b', 'h4', 'alf2'], rparraylist))


# If the program is called from the shell, the following is executed:
#__name__ is the name of the current file
#__main__ is the name of the program that was called from the shell
if __name__ == "__main__":
    
    # Create ArgumentParser object named parser
    parser = argparse.ArgumentParser(description="Calculate confidence intervals for (log(p), p/ptrue, R(M), lambda(M)).")
    
    # Add positional arguments to parser object
    # Files for input data
    # "+" makes a list of the arguments
    parser.add_argument("--file_prefix_list", nargs="+", required=True, help="List of the prefix input files from the emcee runs.")
    parser.add_argument("--file_run_list", nargs="+", required=True, help="List of strings identifying the individual runs.")
    parser.add_argument("--file_suffix", required=True, help="File name after the number string")
    parser.add_argument("--nsystems_list", nargs="+", type=int, required=True, help="List of the number of BNS systems in each run.")
    # Name of EOS
    parser.add_argument("--eos", required=True, help="Name of true eos fit for calculating p/ptrue ['sly', 'eng', 'mpa1', 'ms1', 'ms1b', 'h4', 'alf2'].")
    parser.add_argument("--eos_type", required=True, help="'fit' for piecewise polytrope fit to the tabulate EOS. 'tab' for the tabulated EOS.")
    # Name of output file
    parser.add_argument("--output", required=True, help="Output pickle (.pkl) file.")
    
    # Do the argument parsing
    args = parser.parse_args()
    
    file_prefix_list = args.file_prefix_list
    file_run_list = args.file_run_list
    file_suffix = args.file_suffix
    nsystems_list = args.nsystems_list
    eos = args.eos
    eos_type = args.eos_type
    output_file = args.output
    
    # The eos4paramconfidence module is time consuming, so import it after you have parsed the arguments.
    import eos4parameterpolytrope as eos4p
    import eos4paramconfidence as confidence
    
    # Get list of joined MCMC chains
    eoschainlist = [confidence.load_multiple_emcee_runs(file_prefix, file_run_list, file_suffix, nBurn=50, dSteps=2, flatten=True) for file_prefix in file_prefix_list]
    
    ##### Calculate the confidence contours (log(p), p/ptrue, R(M), lambda(M)) #####
    
    # Calculate log(p) confidence intervals for each simulation
    print('Starting log(p) confidence bound calculations')
    logpboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.log_pressure_confidence_of_log_rho(rhomin, rhomax, nrho, eoschain, conf=conf, method='points')
	print(bounds)
        logpboundslist.append(bounds)
    
    print('Starting p/ptrue confidence bound calculations')
    # Get a table of the EOS for the 'true' (injected) EOS
    if eos_type=='fit':
        # In this case it's the piecewise polytrope least-squares fit to the tabulated EOS
        [lp, g1, g2, g3] = eosfitdict[eos]
        print([lp, g1, g2, g3])
        kTab, gammaTab, rhoTab = eos4p.Set4ParameterPiecewisePolytrope(lp, g1, g2, g3)
        logrhoarray = np.linspace(np.log10(rhomin), np.log10(rhomax), 200)
        rhoarray = 10.0**logrhoarray
        rhoptruearray = np.array([[rho, eos4p.POfRhoPiecewisePolytrope(rho, kTab, gammaTab, rhoTab)] for rho in rhoarray])
    else:
        # In this case it's the tabulated EOS
	print(eos, eos_type)
        print(eostabdict[eos])
        rhoptruearray = eostabdict[eos]
    # Calculate p/ptrue confidence intervals for each simulation
    perrorboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.pressure_error_confidence_of_log_rho(rhomin, rhomax, nrho, rhoptruearray, eoschain, conf=conf, method='points')
	print(bounds)
        perrorboundslist.append(bounds)
    
    
    # Calculate radius confidence intervals for each simulation
    print('Starting radius confidence bound calculations')
    quantity = 'radius'
    radiusboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf, method='points')
	print(bounds)
        radiusboundslist.append(bounds)
    
    
    # Calculate lambda confidence intervals for each simulation
    print('Starting lambda confidence bound calculations')
    quantity = 'lambda'
    lambdaboundslist = []
    for n in range(len(eoschainlist)):
        print(n)
        eoschain = eoschainlist[n]
        bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf, method='points')
	print(bounds)
	lambdaboundslist.append(bounds)

    ## Calculate radius (ignoring BH samples)
    #print('Starting one mode radius confidence bound calculations')
    #quantity = 'radius'
    #radiusmodeboundslist = []
    #for n in range(len(eoschainlist)):
    #    print(n)
    #    eoschain = eoschainlist[n]
    #    bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf, method='restrictmode')
    #	 print(bounds)
    #    radiusmodeboundslist.append(bounds)


    ## Calculate lambda (ignoring BH samples)
    #print('Starting one mode lambda confidence bound calculations')
    #quantity = 'lambda'
    #lambdamodeboundslist = []
    #for n in range(len(eoschainlist)):
    #    print(n)
    #    eoschain = eoschainlist[n]
    #    bounds = confidence.confidence_of_mass(quantity, massmin, massmax, nmass, eoschain, conf=conf, method='restrictmode')
    #	 print(bounds)
    #    lambdamodeboundslist.append(bounds)


    ##################### combine and save data ###################

    #outdict = {'nsystems':nsystems_list, 'logp':logpboundslist, 'perror':perrorboundslist, 'radius':radiusboundslist, 'lambda':lambdaboundslist, 'radiusmode':radiusmodeboundslist, 'lambdamode':lambdamodeboundslist}
    outdict = {'nsystems':nsystems_list, 'logp':logpboundslist, 'perror':perrorboundslist, 'radius':radiusboundslist, 'lambda':lambdaboundslist}
    print(outdict)
     
    # Pickle data
    file = open(output_file, 'w')
    pickle.dump(outdict, file)
    file.close()
