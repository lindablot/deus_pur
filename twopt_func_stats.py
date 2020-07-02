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

    fname = outpath+"/"+output_file_name("twopts", sample, simset, 1, simset.nsimmax, mpole=multipole, irsd=irsd, extension=".npy")
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

    fname = outpath+"/"+output_file_name("mean_twopt", sample, simset, isimmin, isimmax, mpole=multipole, irsd=irsd)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading mean from file: ", fname
        file_content = pd.read_csv(fname, delim_whitespace=True, header=None).values.T
        s = file_content[0]
        twopt_mean = file_content[1]
        twopt_sigma = file_content[2]
    else:
        if okprint:
            print "Computing mean and standard deviation"
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



def cov_twopt(simset, isimmin, isimmax, multipole=0, irsd=0, sample="ndens1", okprint=False, store=False, outpath="."):
    """ Covariance of two point correlation function multipoles

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
    2 numpy arrays
        s, two point function multipole covariance
    """

    fname = outpath+"/"+output_file_name("cov_twopt", sample, simset, isimmin, isimmax, mpole=multipole, irsd=irsd)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading covariance from file: ", fname
        twopt_cov = np.loadtxt(fname,unpack=True)
    else:
        if okprint:
            print "Computing covariance"
        s, twopts = load_twopt(simset, multipole, irsd, sample, okprint, outpath = outpath)
        twopts = twopts[isimmin-1:isimmax]
        twopt_cov=np.cov(twopts,rowvar=False)
        if store:
            if okprint:
                print "Writing file: ", fname
            f = open(fname, "w")
            for i in xrange(0, s.size):
                for j in xrange(0, s.size):
                    f.write(str("%-.12e" % twopt_cov[i, j])+" ")
                f.write("\n")
            f.close()
    return s, twopt_cov



def cross_mpole_cov_twopt(simset, isimmin, isimmax, multipole1, multipole2, irsd=0, sample="ndens1", okprint=False, store=False, outpath="."):
    """ Cross-covariance of two point correlation function between two multipoles

    Parameters
    ----------
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number (default 1)
    isimmax: int
        maximum simulation number
    multipole1: int
        index of first multipole
    multipole2: int
        index of second multipole
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
    2 numpy arrays
        s, two point function multipole cross-covariance
    """

    fname = outpath+"/"+output_file_name("cross_cov_twopt", sample, simset, isimmin, isimmax, mpole=multipole1, mpole2=multipole2, irsd=irsd)
    if os.path.isfile(fname) and not store:
        if okprint:
            print "Reading covariance from file: ", fname
        twopt_cov = np.loadtxt(fname,unpack=True)
    else:
        if okprint:
            print "Computing covariance"
        s, twopts1 = load_twopt(simset, multipole1, irsd, sample, okprint, outpath = outpath)
        s, twopts2 = load_twopt(simset, multipole2, irsd, sample, okprint, outpath = outpath)
        twopts1 = twopts1[isimmin-1:isimmax]
        twopts2 = twopts2[isimmin-1:isimmax]
        mean1 = np.mean(twopts1, axis=0)
        mean2 = np.mean(twopts2, axis=0)
        diff1 = twopts1 - mean1
        diff2 = twopts2 - mean2
        twopt_cov = np.zeros((s.size,s.size))
        for isim in range(isimmin,isimmax):
            twopt_cov += np.outer(diff1[isim-isimmin],diff2[isim-isimmin])
        twopt_cov/=float(isimmax-isimmin)
    return s, twopt_cov



def mpole_cov_twopt(simset, isimmin, isimmax, irsd=0, sample="ndens1", okprint=False, store=False, outpath="."):
    """ Full covariance of the monopole, quadrupole and hexadecapole of the two point correlation function

    Parameters
    ----------
    simset: Simset instance
        simulation set
    isimmin: int
        mimimum simulation number (default 1)
    isimmax: int
        maximum simulation number
    multipole1: int
        index of first multipole
    multipole2: int
        index of second multipole
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
    2 numpy arrays
        s, two point function multipole cross-covariance
    """

    cov_file=outpath+"/"+output_file_name("full_cov_twopt", sample, simset, isimmin, isimmax, mpole=-1, irsd=irsd)
    if os.path.isfile(cov_file) and not store:
        if okprint:
            print "Reading full covariance from file ",cov_file
        full_cov = np.loadtxt(cov_file,unpack=True)
    else:
        s, cov11 = cov_twopt(simset, isimmin, isimmax, 1, irsd, sample, okprint, False, outpath)
        s, cov22 = cov_twopt(simset, isimmin, isimmax, 2, irsd, sample, okprint, False, outpath)
        s, cov33 = cov_twopt(simset, isimmin, isimmax, 3, irsd, sample, okprint, False, outpath)

        s, cov12 = cross_mpole_cov_twopt(simset, isimmin, isimmax, 1, 2, irsd, sample, okprint, False, outpath)
        s, cov13 = cross_mpole_cov_twopt(simset, isimmin, isimmax, 1, 3, irsd, sample, okprint, False, outpath)
        s, cov23 = cross_mpole_cov_twopt(simset, isimmin, isimmax, 2, 3, irsd, sample, okprint, False, outpath)

        full_cov1 = np.concatenate((cov11,cov12,cov13), axis=1)
        full_cov2 = np.concatenate((cov12.T,cov22,cov23), axis=1)
        full_cov3 = np.concatenate((cov13.T,cov23.T,cov33), axis=1)
        full_cov = np.concatenate((full_cov1,full_cov2,full_cov3), axis=0)
        if (store):
            nbin = s.size
            if okprint:
                print "Storing covariance in file ", cov_file
            f = open(cov_file, "w")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov11[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov12[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov13[i,j])+" ")
                f.write("\n")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov12[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov22[i,j])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov23[i,j])+" ")
                f.write("\n")
            for i in xrange(0, nbin):
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov13[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov23[j,i])+" ")
                for j in xrange(0, nbin):
                    f.write(str("%-.12e"%cov33[i,j])+" ")
                f.write("\n")
            f.close()
    return s, full_cov
