# ResTopTag_AN21_019
Resolved top-quark tagger AN2021/019

The repository includes:
* TopTag_* directories that contain:
* Weight files/trained models: weights.h5 
* Scaler file that standardize the data: scaler.save
* Working points: workingPoints.json
* MisID in data and MC - used to estimate the scale factors: topMisID_<WP>.json
* Tagging effic in data and MC - used to estimate the scale factors: topTagEff_genuine_<WP>.json
* Tag/Mistag SF syst unc: toptagEffUncert_genuine_<WP>.json, topmisIDUncert_<WP>.json
* SF for each source of uncertainty topMisID_<WP>_SystVar<SOURCE>.json, topTagEff_<WP>_SystVar<SOURCE>.json - probably not needed as the total variation from the nominal SF is included in the files above. 
* Information about the algorithm (inputVar, scaler, number/type of layers and weights): weights.txt
* An example to implement the model and predict the DNN score is given in "test_main.py" using as input a test rootfile "TopRecoTree_2018_TTMerged_example.root". The output is a root file that contains the histograms of the DNN score for signal (MC-matched) and background (non-matched) top candidates
