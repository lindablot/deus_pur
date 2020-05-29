#!/usr/bin/env python
import numpy as np
from utilities import *

def read_twopt_multi(filename,multipole=0):
    """ Read power spectrum from PowerI4 files.
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

    if multipole==0:
        s, two_pt = np.genfromtxt(filename,unpack=True)
    elif multipole==1:
        s, two_pt, dummy, dummy = np.genfromtxt(filename,unpack=True)
    elif multipole==2:
        s, dummy, two_pt, dummy = np.genfromtxt(filename,unpack=True)
    elif multipole==3:
        s, dummy, dummy, two_pt = np.genfromtxt(filename,unpack=True)
    else:
        raise ValueError("Multipole not implemented")
    return s, two_pt

def twopt_func(simset,nsim,multipole,sample="ndens1",okprint=False):
    if isinstance(simset,Pinocchio10k):
        fname = input_file_name("twopt", simset.mainpath, simset, nsim, powertype=sample)
        s, two_pt = read_twopt_multi(fname,multipole)
    else:
        raise ValueError("Unknown simset class")
    return s, two_pt

def load_twopt(simset,multipole,sample="ndens1",okprint=False,outpath="."):
    fname = outpath+"/"+output_file_name("twopts", sample, simset, 1, 8934, mpole=multipole, extension=".npy")
    if okprint:
        print fname
    s, dummy = twopt_func(simset, 1, multipole, sample)
    if os.path.isfile(fname):
        twopts_array=np.load(fname)
    else:
        twopts_array = np.zeros((8933, s.size))
        for isim in xrange(0, 8933):
            if okprint:
                print simset, isim+1
            dummy, twopts_array[isim] = twopt_func(simset, isim+1, multipole, sample)
        np.save(fname, twopts_array)
    return s, twopts_array

def mean_twopt(simset,isimmin,isimmax,multipole,sample="ndens1",okprint=False,store=False,outpath="."):
    nsim = isimmax-isimmin
    fname = outpath+"/"+output_file_name("mean_twopt", sample, simset, isimmin, isimmax+1, mpole=multipole)

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
        s, twopts = load_twopt(simset, multipole, sample, okprint, outpath = outpath)
        twopts = twopts[isimmin-1:isimmax]
        twopt_mean = np.mean(twopts, axis=0)
        if nsim > 1:
            twopt_sigma = np.std(twopts, axis=0, ddof=1)
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, s.size):
                f.write(str("%-.12e" % s[i])+" "+str("%-.12e"%twopt_mean[i])+" " +
                        str("%-.12e" % twopt_sigma[i])+"\n")
            f.close()

    return s, twopt_mean, twopt_sigma
