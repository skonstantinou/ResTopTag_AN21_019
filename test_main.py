#!/usr/bin/env python
''' 
DESCRIPTION:
This script loads a keras model and predicts the NN output value for a given input dataset


EXAMPLE:
./test_main.py --filename TopRecoTree_2018_TTMerged_example.root --dir TopTag_2018UL --standardise Robust
'''
#================================================================================================ 
# Import modules
#================================================================================================ 
import numpy
import pandas
import keras
import ROOT
import array
import math
import json
import sys
import os
import uproot
import inspect
import joblib
# Regression predictions
from keras.models import load_model
from optparse import OptionParser

# Get parent directory
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Disable AVX/FMA Warning
# Do not display canvases
ROOT.gROOT.SetBatch(ROOT.kTRUE)
# Disable screen output info
ROOT.gROOT.ProcessLine( "gErrorIgnoreLevel = 1001;")

#================================================================================================ 
# Function definition
#================================================================================================ 
def Print(msg, printHeader=False):
    fName = __file__.split("/")[-1]
    if printHeader==True:
        print( "=== ", fName)
        print( "\t", msg)
    else:
        print( "\t", msg)
    return

def main():
    #ROOT.gStyle.SetOptStat(0)

    # Definitions
    filename = opts.filename
    Print("Opening ROOT file %s" %  (opts.filename), True)
    tfile    = ROOT.TFile.Open(filename)    
    sigTree  = "treeS"
    bkgTree  = "treeB"

    #Signal and background branches
    signal     = uproot.open(filename)[sigTree]
    background = uproot.open(filename)[bkgTree]
    
    # List of input variables
    inputList = []
    inputList.append("trijetPtDR")
    inputList.append("dijetPtDR")
    inputList.append("bjetMass")
    inputList.append("LdgJetMass")
    inputList.append("SubldgJetMass")
    inputList.append("trijetMass")
    inputList.append("dijetMass")
    inputList.append("bjetBdisc")
    inputList.append("SoftDrop_n2")
    inputList.append("LdgJetCvsL")
    inputList.append("SubldgJetCvsL")
    inputList.append("bjetCvsL")
    inputList.append("LdgJetPtD")
    inputList.append("SubldgJetPtD")
    inputList.append("LdgJetAxis2")
    inputList.append("SubldgJetAxis2")
    inputList.append("bjetAxis2")
    inputList.append("LdgJetMult")
    inputList.append("SubldgJetMult")
    inputList.append("LdgJetCvsB")
    inputList.append("SubldgJetCvsB")
    inputList.append("bjetCvsB")
    inputList.append("DEtaDijetwithBJet")
    inputList.append("dijetPtOverSumPt")
    inputList.append("LdgJetPtTopCM")
    inputList.append("SubldgJetPtTopCM")
    inputList.append("bjetPtTopCM")
    inputList.append("LdgJetDeltaPtOverSumPt")
    inputList.append("SubldgJetDeltaPtOverSumPt")
    inputList.append("bjetDeltaPtOverSumPt")
    inputList.append("cosW_Jet1Jet2")
    inputList.append("cosW_Jet1BJet")
    inputList.append("cosW_Jet2BJet")
    nInputs = len(inputList)

    # Signal and background dataframe (contain all inputs in TBranches)
    df_signal     = signal.pandas.df(inputList, entrystop=opts.entries)
    df_background = background.pandas.df(inputList, entrystop=opts.entries)
    
    # Concatinate signal + background datasets
    df_all  = pandas.concat( [df_signal, df_background] )

    # Get a Numpy representation of the DataFrames for signal and background datasets
    dset_signal     = df_signal.values
    dset_background = df_background.values
    dataset         = df_all.values

    # Number of events (equal for sig, bkg)
    if opts.entries == None:
            opts.entries = min(len(df_signal.index), len(df_background.index))

    Print("Number of events: %d" % (opts.entries), True)
    
    X_bkg      = dset_background[:opts.entries, 0:nInputs]
    X_sig      = dset_signal[:opts.entries, 0:nInputs]
        
    # Standardization of datasets?
    if opts.standardise != "None":
        msg  = "Standardising dataset features with the %sScaler" % (opts.standardise)
        Print(msg, True)
        scalerName = os.path.join(opts.dir, 'scaler.save')
        # Load the scaler files
        scaler = joblib.load(scalerName)
        
        #Transform inputs
        X_bkg = scaler.transform(X_bkg)
        X_sig = scaler.transform(X_sig)
        
    # Load & compile the model
    modelFile = os.path.join(opts.dir, 'weights.h5')
    Print("Loading model %s" % (modelFile), True)
    loaded_model = load_model(modelFile)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    # Use the loaded model to generate output predictions for the input samples. Computation is done in batches.
    Print("Get the DNN score", True)
    Y_bkg = loaded_model.predict(X_bkg, verbose=opts.verbose) 
    Y_sig = loaded_model.predict(X_sig, verbose=opts.verbose) 
    
    # Create histograms
    h_sig = ROOT.TH1F("sig", '', 50, 0, 1)    
    h_bkg = ROOT.TH1F("bkg", '', 50, 0, 1)    

    #Print("predicted signal: ", False)
    for i in range(opts.entries):
        h_sig.Fill(Y_sig[i])
        #Print(Y_sig[i], False)
    #Print("predicted background: ", False)
    for i in range(opts.entries):
        h_bkg.Fill(Y_bkg[i])
        #Print(Y_bkg[i], False)

    outfile    = ROOT.TFile.Open("ResTopDNN.root","RECREATE")
    outfile.cd()
    h_sig.Write()
    h_bkg.Write()    
    outfile.Close()

    Print("Output: ResTopDNN.root")
#================================================================================================
# Main
#================================================================================================
if __name__ == "__main__":
    '''
    https://docs.python.org/3/library/argparse. html

    name or flags...: Either a name or a list of option strings, e.g. foo or -f, --foo.
    action..........: The basic type of action to be taken when this argument is encountered at the command line.
    nargs...........: The number of command-line arguments that should be consumed.
    const...........: A constant value required by some action and nargs selections.
    default.........: The value produced if the argument is absent from the command line.
    type............: The type to which the command-line argument should be converted.
    choices.........: A container of the allowable values for the argument.
    required........: Whether or not the command-line option may be omitted (optionals only).
    help............: A brief description of what the argument does.
    metavar.........: A name for the argument in usage messages.
    dest............: The name of the attribute to be added to the object returned by parse_args().
    '''
    
    # Default Settings
    FILENAME    = "histograms-TT_19var.root"
    DIR         = None
    SAVEDIR     = ""
    SAVEFORMATS = "pdf" #"png" does not work
    STANDARDISE = "None"
    SAVENAME    = None
    LOGY        = False
    ENTRIES     = None
    VERBOSE     = False

   # Define the available script options
    parser = OptionParser(usage="Usage: %prog [options]")

    parser.add_option("--filename", dest="filename", type="string", default=FILENAME,
                      help="Input ROOT file containing the signal and backbground TTrees with the various TBranches *variables) [default: %s]" % FILENAME)

    parser.add_option("--standardise", dest="standardise", default=STANDARDISE,
                      help="Standardizing a dataset involves rescaling the distribution of INPUT values so that the mean of observed values is 0 and the standard deviation is 1 (e.g. StandardScaler) [default: %s]" % STANDARDISE)

    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=VERBOSE,
                      help="Enable verbose mode (for debugging purposes mostly) [default: %s]" % VERBOSE)

    parser.add_option("--entries", dest="entries", type=int, default=ENTRIES,
                      help="Number of entries to be used in filling the mass histogram [default: %s]" % ENTRIES)

    parser.add_option("--dir", dest="dir", default=DIR,
                      help="Directory where the model training file is located (\"weights.h5\") [default: %s]" % DIR)


    (opts, parseArgs) = parser.parse_args()

    
    # Sanity check
    if opts.dir == None:
        raise Exception("No directory was defined where the model training file is located (\"weights.h5\")! Please define an input directory with the --dir option")

    # is the input standardised?
    cfgFile = os.path.join(opts.dir, "config.json")
    if os.path.exists(cfgFile):
        f = open(cfgFile, "r")
        config = json.load(f)
        f.close()
        if "standardised datasets" in config:
            if config["standardised datasets"] != "None":
                opts.standardise = (config["standardised datasets"])

    main()
