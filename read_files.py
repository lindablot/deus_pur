#!/usr/bin/env python
# read_files.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import numpy as np
import pandas as pd
from utilities import *
# ---------------------------------------------------------------------------- #


# -------------------------------- READ DATA --------------------------------- #
def read_cosmo_data(mainpath="", simset=DeusPurSet("all_256")):
    """ Read linear power spectrum and lookup tables for cosmological quantities 
    
    Parameters
    ---------
    mainpath : str
        path to base folder (default is empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
        
    Returns
    -------
    numpy array
        k values of linear power spectrum
    numpy array
        linear power spectrum
    numpy array
        expansion factor values of growth factor
    numpy array
        growth factor
    numpy array
        expansion factor values of hubble parameter and proper time
    numpy array
        hubble parameter
    numpy array
        proper time
    """
    
    model = simset.model
    file_content = pd.read_csv(mainpath+"/data/pk_"+model+".dat", header=None, delim_whitespace=True).values
    power_k = file_content.T[0]
    power_p = file_content.T[1]
    file_content = pd.read_csv(mainpath+"/data/mpgrafic_input_"+model+".dat", header=None, delim_whitespace=True).values
    growth_a = file_content.T[0]
    growth_dplus = file_content.T[2]
    file_content = pd.read_csv(mainpath+"/data/ramses_input_"+model+".dat", header=None, delim_whitespace=True).values
    evolution_a = file_content.T[0]
    evolution_hh0 = file_content.T[1]
    evolution_tproperh0 = file_content.T[4]
    return power_k, power_p, growth_a, growth_dplus, evolution_a, evolution_hh0, evolution_tproperh0
# ---------------------------------------------------------------------------- #


# -------------------------------- READ CAMB POWER SPECTRUM ------------------ #
def read_power_camb(mainpath="", simset=DeusPurSet("all_256")):
    """ Read linear power spectrum
    
    Parameters
    ---------
    mainpath : str
        Path to base folder (default is empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
        
    Returns
    -------
    numpy array
        k values of linear power spectrum
    numpy array
        linear power spectrum
    """
    
    model = simset.model
    file_content = pd.read_csv(mainpath+"/data/pk_"+model+".dat", header=None, delim_whitespace=True).values
    power_k = file_content.T[0]
    power_p = file_content.T[1]
    return power_k, power_p
# ---------------------------------------------------------------------------- #


# -------------------------------- READ GROWTH FACTOR ------------------------ #
def read_growth(mainpath="", simset=DeusPurSet("all_256")):
    """ Read growth factor
    
    Parameters
    ---------
    mainpath : str
        Path to base folder (default is empty)
    simset: Simset instance
        simulation set (default DeusPurSet("all_256"))
        
    Returns
    -------
    numpy array
        expansion factor values of growth factor
    numpy array
        growth factor
    """
    
    model = simset.model
    file_content = pd.read_csv(mainpath+"/data/mpgrafic_input_"+model+".dat", header=None, delim_whitespace=True).values
    growth_a = file_content.T[0]
    growth_dplus = file_content.T[2]
    return growth_a, growth_dplus
# ---------------------------------------------------------------------------- #


# -------------------------------- READ POWER -------------------------------- #
def read_power_powergrid(filename="", refcol=np.zeros(0)):
    """ Read powergrid output file
    
    Parameters
    ---------
    filename : str
        Name of powergrid output file (default is empty)
        
    Returns
    -------
    numpy array
        k values of power spectrum
    numpy array
        power spectrum values
    numpy array
        some strange function of the expansion factor
    """
    
    file_content = pd.read_csv(filename, header=None, delim_whitespace=True).values
    data1 = file_content.T[0]
    data2 = file_content.T[1]
    data3 = file_content.T[2]
    if refcol.size > 0:
        column1 = np.array(refcol)
        column2 = np.zeros(refcol.size)
        column3 = np.zeros(refcol.size)
        counter = np.zeros(refcol.size)
        i = data1.size-1
        j = refcol.size-1
        while True:
            if np.log(data1[i]) > (np.log(refcol[j])+np.log(refcol[j-1]))/2.:
                counter[j] += 1
                column2[j] += data2[i]
                column3[j] += data3[i]
                i -= 1
            elif j > 1:
                j -= 1
            else:
                break
        for i in xrange(0, refcol.size):
            if counter[i] > 0.:
                column2[i] /= counter[i]
                column3[i] /= counter[i]
    else:
        column1 = data1
        column2 = data2
        column3 = data3
    return column1, column2, column3
# ---------------------------------------------------------------------------- #



# ------------------------------ READ POWERI4 -------------------------------- #
def read_power_powerI4(filename="",multipole=0,bin_center=False):
    """ Read power spectrum from PowerI4 files.
    Multipoles are:
    0: real space
    1: monopole
    2: quadrupole
    3: hexadecapole
    
    Parameters
    ----------
    filename: string
        name of the file (default empty)
    multipole: int
        multipole index (default 0)
    bin_center: bool
        center of k bin or average k of the bin (default False)
    
    Returns
    -------
    3 numpy arrays
        k, pk and number of modes
    """
    
    if multipole==0:
        power_k1, power_k2, power_p, nmodes = np.genfromtxt(filename,skip_header=1,unpack=True)
    elif multipole==1:
        power_k1, power_k2, power_p, dummy, dummy, nmodes = np.genfromtxt(filename,skip_header=1,unpack=True)
    elif multipole==2:
        power_k1, power_k2, dummy, power_p, dummy, nmodes = np.genfromtxt(fname,skip_header=1,unpack=True)
    elif multipole==3:
        power_k1, power_k2, dummy, dummy, power_p, nmodes = np.genfromtxt(fname,skip_header=1,unpack=True)
    else:
        raise ValueError("Multipole not implemented")
    if bin_center:
        return power_k2, power_p, nmodes
    else:
        return power_k1, power_p, nmodes
# ---------------------------------------------------------------------------- #



# ---------------------------- READ OBJECT NUMBER ---------------------------- #
def read_nobjects(filename=""):
    """ Read object number from PowerI4 file
    
    Parameters
    ----------
    filename: string
        name of the file (default empty)
        
    Returns
    -------
    int
        number of objects
    """
    
    with open(fname, 'r') as f:
        first_line = f.readline().strip()
        nobjects=int(first_line.split()[0])
    return nobjects
# ---------------------------------------------------------------------------- #



# -------------------------------- READ INFO --------------------------------- #
def read_aexp_info(filename=""):
    """ Extract exact expansion factor value from the simulation info file
    
    Parameters
    ---------
    filename : str
        Name of simulation info file (default is empty)
        
    Returns
    -------
    double
        expansion factor
    """
    
    aexp = 0
    lines = [line.strip() for line in open(filename)]
    for i in xrange(0, len(lines)):
        if lines[i].count("aexp") > 0:
            words = lines[i].split("=")
            aexp = float(words[1].strip())
        elif aexp != 0:
            break
    return aexp
# ---------------------------------------------------------------------------- #


# ---------------------------- READ MASSFUNCTION ----------------------------- #
def read_massfunction(filename=""):
    """ Read halo mass function file
    
    Parameters
    ---------
    filename
        Name of mass function file (default is empty)
    
    Returns
    -------
    numpy array
        Left edges of mass bins
    numpy array
        Right edges of mass bins
    numpy array
        Halo counts in mass bins
    """
    
    mf_binmin, mf_binmax, mf_count = np.loadtxt(filename, unpack=True)
    file_content = pd.read_csv(filename, " ", header=None).values
    mf_binmin = file_content[0]
    mf_binmax = file_content[1]
    mf_count = file_content[2]
    return mf_binmin, mf_binmax, mf_count
# ---------------------------------------------------------------------------- #
