import argparse
import numpy as np
import eos4parameterpolytrope as eos4p

# If the program is called from the shell, the following is executed:
#__name__ is the name of the current file
#__main__ is the name of the program that was called from the shell
if __name__ == "__main__":
    
    # Create ArgumentParser object named parser
    parser = argparse.ArgumentParser(description="Estimate the 4 EOS parameters from the first nSystems BNS events using results from lalinference_mcmc for each event.")
    
    # Add positional arguments to parser object
    # Files for input data
    # "+" makes a list of the arguments
    parser.add_argument("files", nargs="+", metavar="mcmcinputfile", help="List of output files from lalinference_mcmc runs.")
    # Name of output file
    parser.add_argument("output", help="Output file for MCMC results. Must be type .npy.")
    
    # Add optional arguments of type int to the parser object
    #parser.add_argument("-n", "--nsystems", type=int, default=1, help="Number of systems.")
    parser.add_argument("-w", "--walkers", type=int, default=100, help="Number of walkers in MCMC simulation.")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps to take for each walker in MCMC simulation.")
    parser.add_argument("-d", "--dsteps", type=int, default=100, help="Save every dth step so output file isn't huge. Should be ~autocorrelation length.")
    parser.add_argument("-m", "--massknown", type=float, default=1.93, help="Maximum known NS mass.")
    parser.add_argument("-v", "--maxvs", type=float, default=1.0, help="Maximum allowed speed of sound (geometric c=1 units).")
    #Currently only accepts single thread for MCMC input instead of Fisher matrix input
    #parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads to use.")
    
    # Do the argument parsing
    args = parser.parse_args()
    
    print(args.files)
    print(args.output)
    
    filepathlist = args.files
    nWalkers = args.walkers
    nBurn = 100
    nSteps = args.steps
    dSteps = args.dsteps
    massKnown = args.massknown
    causalLimit = args.maxvs
    # Run the MCMC simulation
    chain = eos4p.RunMCMCFor4ParamEOSWithLikeFromMCMC(filepathlist, nWalkers, nBurn, nSteps, massKnown=massKnown, causalLimit=causalLimit)
    
    # Save only the 4 EOS parameters to a .npy file
    np.save(args.output, chain[:, ::dSteps, [0, 1, 2, 3]])
