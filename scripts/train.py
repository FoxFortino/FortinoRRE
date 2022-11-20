import os
from os.path import join, isdir
import sys

sys.path.insert(0, "/home/2649/repos/TRANSSClass/scripts")
import TRANSSC_utils as tutils
import deep_learning_multilayer as dlm

def main(R):
    
    # The directory where training_params.zip will be created
    # The directory where the training set will be created
    # The directory where the DASH model training will take place
    # If it does not exist, simply make the empty directory
    R_dir = join("/lustre/lrspec/users/2649/models/FortinoRRE", str(R))
    if not isdir(R_dir):
        os.mkdir(R_dir)
    os.chdir(R_dir)
    
    BACKUP_DIR = os.path.join(R_dir, "backup")
    if not os.path.isdir(BACKUP_DIR):
        os.mkdir(BACKUP_DIR)
        
    dlm.train_model(
        R_dir,
        num_epochs=100,
        batch_size=32,
        restart_fit=False,
        BACKUP_DIR=BACKUP_DIR)

if __name__ == "__main__":
    R = sys.argv[1]
    main(R)
