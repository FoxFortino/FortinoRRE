import sys
import os
from os.path import join, isdir

sys.path.insert(0, "/home/2649/repos/TRANSSClass/scripts")
import TRANSSC_utils as tutils
import degrade

import training_parameters as tp
import create_training_set as cts


LIBRARY_PATH = "/lustre/lrspec/users/2649/FortinoRRE_spectralib/"

def main(R):
    
    # Create library of degraded spectra at LIBRARY_PATH/R/lnw_files
    degrade.degrade_all(R, tutils.NO_DUPE, LIBRARY_PATH, print_info=False)
    
    # The path where the lnw files for each SN are kept
    DATA_dir = join(LIBRARY_PATH, str(R), "lnw_files")
    assert isdir(DATA_dir)
    
    # The directory where training_params.zip will be created
    # The directory where the training set will be created
    # The directory where the DASH model training will take place
    # If it does not exist, simply make the empty directory
    R_dir = join("/lustre/lrspec/users/2649/models/FortinoRRE", str(R))
    if not isdir(R_dir):
        os.mkdir(R_dir)

    # Make the training params file in R_dir
    os.chdir(R_dir)
    tp.create_training_params_file(R_dir)
    
    # Generate the training/testing set in R_dir
    trainingSetFilename = cts.create_training_set_files(
        R_dir,
        snidTemplateLocation=DATA_dir)


if __name__ == "__main__":
    R = int(sys.argv[1])
    main(R)
