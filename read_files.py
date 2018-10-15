#!/usr/bin/env python
# read_files.py - Vincent Reverdy (vince.rev@gmail.com) and Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import numpy as np
# ---------------------------------------------------------------------------- #


# -------------------------------- READ DATA --------------------------------- #
def read_data(mainpath="", model="lcdmw7"):
    """ Read linear power spectrum and lookup tables for cosmological quantities 
    
    Parameters
    ---------
    mainpath : str
        Path to the data folder (default is empty)
    model : str, optional
        Cosmological model (default is lcdmw7)
        
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
    
    power_k, power_p = np.loadtxt(mainpath+"/data/pk_"+model+".dat", unpack=True)
    growth_a, dummy, growth_dplus, dummy = np.loadtxt(mainpath+"/data/mpgrafic_input_"+model+".dat", unpack=True)
    evolution_a, evolution_hh0, dummy, dummy, evolution_tproperh0 = \
        np.loadtxt(mainpath+"/data/ramses_input_"+model+".dat", unpack=True)
    return power_k, power_p, growth_a, growth_dplus, evolution_a, evolution_hh0, evolution_tproperh0
# ---------------------------------------------------------------------------- #


# -------------------------------- READ CAMB POWER SPECTRUM ------------------ #
def read_power_camb(mainpath="", model="lcdmw7"):
    """ Read linear power spectrum
    
    Parameters
    ---------
    mainpath : str
        Path to the data folder (default is empty)
    model : str, optional
        Cosmological model (default is lcdmw7)
        
    Returns
    -------
    numpy array
        k values of linear power spectrum
    numpy array
        linear power spectrum
    """
    
    power_k, power_p = np.loadtxt(mainpath+"/data/pk_"+model+".dat", unpack=True)
    return power_k, power_p
# ---------------------------------------------------------------------------- #


# -------------------------------- READ GROWTH FACTOR ------------------------ #
def read_growth(mainpath="", model="lcdmw7"):
    """ Read growth factor
    
    Parameters
    ---------
    mainpath : str
        Path to the data folder (default is empty)
    model : str, optional
        Cosmological model (default is lcdmw7)
        
    Returns
    -------
    numpy array
        expansion factor values of growth factor
    numpy array
        growth factor
    """
    
    growth_a, dummy, growth_dplus, dummy = np.loadtxt(mainpath+"/data/mpgrafic_input_"+model+".dat", unpack=True)
    return growth_a, growth_dplus
# ---------------------------------------------------------------------------- #


# -------------------------------- READ POWER -------------------------------- #
def read_power(filename="", refcol=np.zeros(0)):
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
    
    data1, data2, data3 = np.loadtxt(filename, unpack=True)
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


# -------------------------------- READ INFO --------------------------------- #
def read_info(filename=""):
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
    return mf_binmin, mf_binmax, mf_count
# ---------------------------------------------------------------------------- #
