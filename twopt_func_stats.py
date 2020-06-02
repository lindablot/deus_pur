#!/usr/bin/env python
import numpy as np
from utilities import *

def read_twopt_multi(filename, multipole=0):
    """ Read two point correlation function multipole file.
    Multipoles are:
    0: real space
    1: monopole
    2: quadrupole
    3: hexadecapole

    Parameters
    ----------
    filename: string
        name of the file
    multipole: int
        multipole index (default 0)

    Returns
    -------
    2 numpy arrays
        s, two point function multipole
    """

    if multipole<=1:
        s, two_pt, dummy, dummy = np.genfromtxt(filename,unpack=True)
    elif multipole==2:
        s, dummy, two_pt, dummy = np.genfromtxt(filename,unpack=True)
    elif multipole==3:
        s, dummy, dummy, two_pt = np.genfromtxt(filename,unpack=True)
    else:
        raise ValueError("Multipole not implemented")
    return s, two_pt

def twopt_func(simset, nsim, multipole=0, irsd=0, sample="ndens1", okprint=False):
    """ Two point correlation function multipole
    Multipoles are:
    0: real space
    1: monopole
    2: quadrupole
    3: hexadecapole

    Parameters
    ----------
    simset: Simset instance
        simulation set
    multipole: int
        multipole index (default 0)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    sample: str
        sample name (default "ndens1")
    okprint: bool
        verbose (default False)

    Returns
    -------
    2 numpy arrays
        s, two point function multipole
    """

    if irsd==0 and multipole!=0:
        raise ValueError("When reading multipole files you need to specify irsd between 1 and 3")
    if isinstance(simset,Pinocchio10k):
        fname = input_file_name("twopt", simset.mainpath, simset, nsim, powertype=sample, irsd=irsd)
        s, two_pt = read_twopt_multi(fname, multipole)
    else:
        raise ValueError("Unknown simset class")
    return s, two_pt

def load_twopt(simset, multipole=0, irsd=0, sample="ndens1", okprint=False, outpath="."):
    """
    Load all two point correlation function of a given simulation set in memory

    Parameters
    ----------
    simset: Simset instance
        simulation set
    noutput: int
         snapshot number
    multipole: int
        multipole index (default 0)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    okprint: bool
        verbose (default False)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    2 numpy arrays
        vector of s values and array of two point correlation functions of shape (nsim,nbin)
    """

    fname = outpath+"/"+output_file_name("twopts", sample, simset, 1, simset.nsimmax+1, mpole=multipole, irsd=irsd, extension=".npy")
    if okprint:
        print fname
    s, dummy = twopt_func(simset, 1, multipole, irsd, sample)
    if os.path.isfile(fname):
        twopts_array=np.load(fname)
    else:
        twopts_array = np.zeros((simset.nsimmax, s.size))
        for isim in xrange(0, simset.nsimmax):
            try:
                dummy, twopts_array[isim] = twopt_func(simset, isim+1, multipole, irsd, sample)
            except:
                twopts_array[isim,:] = np.nan
                if okprint:
                    print "File number ",isim," is missing"
        np.save(fname, twopts_array)
    return s, twopts_array

def mean_twopt(simset, isimmin, isimmax, multipole=0, irsd=0, sample="ndens1", okprint=False, store=False, outpath="."):
    """ Mean and standard deviation of two point correlation function multipoles

    Parameters
    ----------
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number (default 1)
    isimmax: int
        maximum simulation number
    multipole: int
        multipole index (default 0)
    irsd: int
        direction of redshift space index (0 for real space, 1-3 for x,y or z, default 0)
    sample: str
        sample name (default "ndens1")
    okprint: bool
        verbose (default False)
    store: bool
        store file. If True and file exists it will be overwritten (default False)
    outpath: string
        path where output file is stored (default .)

    Returns
    -------
    3 numpy arrays
        s, two point function multipole mean and standard deviation
    """

    fname = outpath+"/"+output_file_name("mean_twopt", sample, simset, isimmin, isimmax+1, mpole=multipole, irsd=irsd)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading mean spectrum from file: ", fname
        file_content = pd.read_csv(fname, delim_whitespace=True, header=None).values.T
        s = file_content[0]
        twopt_mean = file_content[1]
        twopt_sigma = file_content[2]
    else:
        if okprint:
            print "Computing mean and standard deviation of spectra"
        s, twopts = load_twopt(simset, multipole, irsd, sample, okprint, outpath = outpath)
        twopts = twopts[isimmin-1:isimmax]
        twopt_mean = np.nanmean(twopts, axis=0)
        twopt_sigma = np.nanstd(twopts, axis=0, ddof=1)
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, s.size):
                f.write(str("%-.12e" % s[i])+" "+str("%-.12e"%twopt_mean[i])+" " +
                        str("%-.12e" % twopt_sigma[i])+"\n")
            f.close()
    return s, twopt_mean, twopt_sigma
