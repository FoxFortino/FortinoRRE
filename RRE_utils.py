import sys
import os
import glob
import itertools

import numpy as np
import pandas as pd

"""Specify useful directories that aren't already specified in PYTHONPATH"""
HOME_DIR = "/home/2649/"
REPO_DIR = os.path.join(HOME_DIR, "repos")

SESNspectraPCA_DIR = os.path.join(REPO_DIR, "SESNspectraPCA")
SESNspectraLib_DIR = os.path.join(REPO_DIR, "SESNspectraLib")
SESNtemple_DIR = os.path.join(REPO_DIR, "SESNtemple")
dgspec_DIR = os.path.join(REPO_DIR, "dgspec")
adfox_DIR = os.path.join(REPO_DIR, "adfox")

adfox_CODE = os.path.join(adfox_DIR, "astrodash")
SESNspectraPCA_CODE = os.path.join(SESNspectraPCA_DIR, "code")

sys.path.insert(0, SESNspectraPCA_DIR)
sys.path.insert(0, SESNspectraLib_DIR)
sys.path.insert(0, SESNtemple_DIR)
sys.path.insert(0, dgspec_DIR)
sys.path.insert(0, adfox_DIR)

sys.path.insert(0, adfox_CODE)
sys.path.insert(0, SESNspectraPCA_CODE)

"""Gather lists of the data files for this research"""
adfox_DATA = glob.glob(
    os.path.join(adfox_DIR, "templates/training_set/*lnw")
)
SESNtemple_DATA = glob.glob(
    os.path.join(SESNtemple_DIR, "SNIDtemplates/templates_*/*lnw")
)
SESNtemple_DATA1 = glob.glob(
    os.path.join(SESNtemple_DIR, "templates_adjusted/*lnw")
)
SESNtemple_DATA2 = glob.glob(
    os.path.join(SESNtemple_DIR, "templates_new/*lnw")
)
SESNtemple_DATA3 = glob.glob(
    os.path.join(SESNtemple_DIR, "templates_williamson/*lnw")
)

# Loop through all files to look for duplicates between the SESN data and the
# astrodash data. If there is a duplicate, keep the SNID one in the list
# for no particular reason other than it is probably more updated.
ALL_DATA = adfox_DATA + SESNtemple_DATA
ALL_LNW = [os.path.basename(file) for file in ALL_DATA]
NO_DUPE = []
for lnw, file in zip(ALL_LNW, ALL_DATA):
    if file in SESNtemple_DATA:
        NO_DUPE.append(file)

    elif file in adfox_DATA:
        if lnw in [os.path.basename(f) for f in SESNtemple_DATA]:
            continue
        else:
            NO_DUPE.append(file)


"""Initialize useful dictionaries for SN data."""
# List of 17 SN subtypes as string and int.
SNtypes_str = np.array(
    [
        "Ia-norm", "Ia-91T", "Ia-91bg", "Ia-csm", "Iax", "Ia-pec",
        "Ib-norm", "Ibn", "IIb", "Ib-pec",
        "Ic-norm", "Ic-broad", "Ic-pec",
        "IIP", "IIL", "IIn", "II-pec"
    ]
)
SNtypes_int = np.arange(SNtypes_str.size)

# Dictionaries for converting between SN subtype int <-> str.
SNtypes_str_to_int = {
    SNstr: SNint for SNstr, SNint in zip(SNtypes_str, SNtypes_int)
}
SNtypes_int_to_str = {
    j: i for i, j in SNtypes_str_to_int.items()
}

# List of 4 SN broadtypes as string and int.
SNbroadtypes_str = np.array(["Ia", "Ib", "Ic", "II"])
SNbroadtypes_int = np.arange(SNbroadtypes_str.size)

# Dictionaries for converting between SN broadtype int <-> str.
SNbroadtypes_str_to_int = {
    SNstr: SNint for SNstr, SNint in zip(SNbroadtypes_str, SNbroadtypes_int)
}
SNbroadtypes_int_to_str = {
    j: i for i, j in SNbroadtypes_str_to_int.items()
}


"""Common corrections for SN names"""
SNtypes_str_to_int["Ia-02cx"] = SNtypes_str_to_int["Iax"]
SNtypes_str_to_int["Ia-99aa"] = SNtypes_str_to_int["Ia-91T"]
SNtypes_str_to_int["Ib"] = SNtypes_str_to_int["Ib-norm"]
SNtypes_str_to_int["Ic"] = SNtypes_str_to_int["Ic-norm"]


"""Useful functions for SN data"""
@np.vectorize
def get_broad_type(SNtype_int, return_str=False):
    """
    Given an integer representing a SN subtype, find its broadtype.
    
    For this work, we considered 17 different SN types. In this script you can
    find which SN types they are and which integer each type corresponds to.
    This function takes that integer and returns the broad type of the SN
    (either Ia, Ib, Ic, or II).
    
    Arguments
    ---------
    SNtype_int : int or iterable of ints
        An integer corresponding to a SN type, or a list of such integers.
    
    Keyword Arguments
    -----------------
    return_str : bool, Default: False
        If True, return the broad type as string. If false, return an integer
        (either 0, 1, 2, or 3).
        
    Returns
    -------
    broadtype : int or str
        The integer (or string, if return_str = True) representing the broad
        type of the subtype, SNtype_int, that was given.
    """
    if 0 <= SNtype_int <= 5:
        val = "Ia"
    elif 6 <= SNtype_int <= 9:
        val = "Ib"
    elif 10 <= SNtype_int <= 12:
        val = "Ic"
    elif 13 <= SNtype_int <= 16:
        val = "II"

    if not return_str:
        return SNbroadtypes_str_to_int[val]
    else:
        return val


def agetype_to_subtype(agetypes, typeNamesList, nAges=18):
    """
    
    """
    bins = np.arange(0, typeNamesList.size, nAges)
    subtypes = np.digitize(agetypes, bins) - 1
    return subtypes

    
def redshift(flux, wvl0, z, blue=True):
    if blue:
        wvl = wvl0 / (1 + z)
    else:
        wvl = wvl0 * (1 + z)
    redshifted_flux = np.interp(wvl0, wvl, flux, left=0, right=0)
    return redshifted_flux