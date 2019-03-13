#!/usr/bin/env python
# wishart_analysis.py - Linda Blot (linda.blot@obspm.fr) - 2014
# ---------------------------------- IMPORT ---------------------------------- #
import math
import os
import numpy as np
from numpy import linalg
from read_files import *
from power_types import *
from power_stats import *
from power_covariance import *
# ---------------------------------------------------------------------------- #



# --------------------------- COVARIANCE VARIANCE --------------------------- #
def cov_stats(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint = False, store = True):
    """ Mean and standard deviation of covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 2 numpy arrays
        list of realization numbers, mean and standard deviation of covariance
    """
    
    simset = DeusPurSet("all_256")
    i=0 #index on list_nr
    
    if store:
        folder="tmp/"+str("%05d"%noutput)+"/"
        command="mkdir -p tmp; mkdir -p "+folder
        os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    nbin = power_k.size
    
    cov_mean = np.zeros((len(list_nr),nbin,nbin))
    cov_sigma = np.zeros((len(list_nr),nbin,nbin))
    
    for nr in list_nr:
        
        nsub = 12288/nr
        cov_sub = np.zeros((nsub,nbin,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%power_pcov[ik,jk])+" ")
                        f.write("\n")
                    f.close()
            
            cov_sub[isub]=power_pcov
    
        cov_mean[i] = np.mean(cov_sub,axis=0)
        cov_sigma[i] = np.std(cov_sub,axis=0,ddof=1)

        i+=1
    return list_nr, cov_mean, cov_sigma
# ---------------------------------------------------------------------------- #



# --------------------------- TRACE OF SAMPLE COVARIANCE VARIANCE --------------------------- #
def cov_variance_trace(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint = False, store = True):
    """ Trace of the relative variance of the covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers, trace of relative covariance variance
    """
    
    simset = DeusPurSet("all_256")
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    if store:
        folder="tmp/"+str("%05d"%noutput)+"/"
        command="mkdir -p tmp; mkdir -p "+folder
        os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    nbin = power_k.size
    
    for nr in list_nr:
        
        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        nsim = 0
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"sigma_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            filename_cov=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_psigma=np.loadtxt(filename, unpack=True)
            elif(os.path.isfile(filename_cov)):
                if (okprint):
                    print "Reading file: ",filename_cov
                power_pcov=np.loadtxt(filename_cov, unpack=True)
                power_psigma=np.sqrt(np.diag(power_pcov))
            else:
                dummy,dummy,power_psigma=mean_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        f.write(str(fltformat%power_psigma[ik])+"\n")
                    f.close()
    
            power_pvar[isub]=power_psigma**2
        

        var_mean=np.mean(power_pvar,axis=0)
        sigma=np.std(power_pvar,axis=0,ddof=1)
        sigma2=sigma*sigma
        fact=np.sum(var_mean*var_mean)
        trace_sigma2[i]=np.sum(sigma2)/fact
        
        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- TRACE OF SAMPLE COVARIANCE VARIANCE WITH k CUT --------------------------- #
def cov_variance_trace_kcut(kmin=0.03, kmax = 1., powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Trace of the relative variance of the covariance estimator using all_256 Deus Pur set cut at given k_min and k_max.
        
    Paramters
    ---------
    kmin: float
        minimum k (default 0.03 h/Mpc)
    kmax: float
        maximum k (default 1 h/Mpc)
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and trace of relative covariance variance
    """
    
    simset = DeusPurSet("all_256")
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    index = [(power_k > kmin) & (power_k < kmax)]
    power_k_cut = power_k[index]
    nbin = power_k_cut.size
    
    for nr in list_nr:
        
        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        nsim = 0
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"sigma_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            filename_cov=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_psigma=np.loadtxt(filename, unpack=True)
            elif(os.path.isfile(filename_cov)):
                if (okprint):
                    print "Reading file: ",filename_cov
                power_pcov=np.loadtxt(filename_cov, unpack=True)
                power_psigma=np.sqrt(np.diag(power_pcov))
            else:
                dummy,dummy,power_psigma=mean_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, power_k.size):
                        f.write(str(fltformat%power_psigma[ik])+"\n")
                    f.close()
    
            power_pvar[isub]=(power_psigma[index])**2
    
        var_mean=np.mean(power_pvar,axis=0)
        sigma=np.std(power_pvar,axis=0,ddof=1)
        sigma2=sigma*sigma
        fact=np.sum(var_mean*var_mean)
        trace_sigma2[i]=np.sum(sigma2)/fact
    
        i+=1

    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- COVARIANCE VARIANCE --------------------------- #
def cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Relative variance of the covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and relative covariance variance
    """
    
    simset = DeusPurSet("all_256")
    i=0 #index on list_nr
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    nbin = power_k.size
    
    sigma2_all = np.zeros((len(list_nr),nbin,nbin))
    
    for nr in list_nr:
        
        nsub = 12288/nr
        cov_sub = np.zeros((nsub,nbin,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%power_pcov[ik,jk])+" ")
                        f.write("\n")
                    f.close()
            
            cov_sub[isub]=power_pcov
        
        cov_mean = np.mean(cov_sub,axis=0)
        sigma = np.std(cov_sub,axis=0,ddof=1)
        sigma2 = sigma*sigma
        fact = np.outer(np.diag(cov_mean),np.diag(cov_mean))+cov_mean*cov_mean
        
        sigma2_all[i]=sigma2/fact

        i+=1
    return list_nr, sigma2_all
# ---------------------------------------------------------------------------- #



# --------------------------- COVARIANCE VARIANCE WITH k CUT --------------------------- #
def cov_variance_kcut(kmin=0.03, kmax = 1., powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Relative variance of the covariance estimator using all_256 Deus Pur set cut at given k_min and k_max.
        
    Paramters
    ---------
    kmin: float
        minimum k (default 0.03 h/Mpc)
    kmax: float
        maximum k (default 1 h/Mpc)
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and relative covariance variance
    """
    
    simset = DeusPurSet("all_256")
    i=0 #index on list_nr
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    ikmin = np.searchsorted(power_k,kmin)
    ikmax = np.searchsorted(power_k,kmax)
    power_k_cut = power_k[ikmin:ikmax]
    nbin = power_k_cut.size
    
    sigma2_all = np.zeros((len(list_nr),nbin,nbin))
    
    for nr in list_nr:
        
        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        cov_sub = np.zeros((nsub,nbin,nbin))
        nsim = 0
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%power_pcov[ik,jk])+" ")
                        f.write("\n")
                    f.close()
            
            cov_sub[isub] = power_pcov[ikmin:ikmax,ikmin:ikmax]
        
        cov_mean = np.mean(cov_sub,axis=0)
        sigma = np.std(cov_sub,axis=0,ddof=1)
        sigma2 = sigma*sigma
        fact = np.outer(np.diag(cov_mean),np.diag(cov_mean))+cov_mean*cov_mean
        
        sigma2_all[i]=sigma2/fact
        
        i+=1
    return list_nr, sigma2_all
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX VARIANCE TRACE --------------------------- #
def inv_cov_variance_trace(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [300,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Trace of the relative variance of the inverse of the covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute inverse covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers, trace of inverse covariance relative variance
    """

    simset = DeusPurSet("all_256")
    trace_sigma2=np.zeros(len(list_nr))
    i=0
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)

    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    nbin = power_k.size
    var_all = np.zeros(nbin)
    var_inv_mean = np.zeros(nbin)
    
    for nr in list_nr:
        nsub = 12288/nr
        var_inv = np.zeros((nsub,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))

        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                cov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%cov[ik,jk])+" ")
                        f.write("\n")
                    f.close()

            var_inv[isub]=np.diag(linalg.inv(cov))

        var_inv_mean = np.mean(var_inv,axis=0)
        sigma = np.std(var_inv,axis=0,ddof=1)
        sigma2 = sigma*sigma
        fact=np.sum(var_inv_mean*var_inv_mean)
        
        trace_sigma2[i]=np.sum(sigma2)/fact

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX VARIANCE TRACE WITH k CUT --------------------------- #
def inv_cov_variance_trace_kcut(kmin = 0.03, kmax = 1., powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [300,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Trace of the relative variance of the inverse of the covariance estimator using all_256 Deus Pur set cut at given k_min and k_max.
        
    Paramters
    ---------
    kmin: float
        minimum k (default 0.03 h/Mpc)
    kmax: float
        maximum k (default 1 h/Mpc)
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute inverse covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and trace of inverse covariance relative variance
    """
    
    simset = DeusPurSet("all_256")
    trace_sigma2=np.zeros(len(list_nr))
    i=0
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    ikmin = np.searchsorted(power_k,kmin)
    ikmax = np.searchsorted(power_k,kmax)
    power_k_cut = power_k[ikmin:ikmax]
    nbin = power_k_cut.size
    
    for nr in list_nr:
        nsub = 12288/nr
        var_inv = np.zeros((nsub,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim
        
        var_all = np.zeros(nbin)
        var_inv_mean = np.zeros(nbin)
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                cov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, power_k.size):
                        for jk in xrange(0, power_k.size):
                            f.write(str(fltformat%cov[ik,jk])+" ")
                        f.write("\n")
                    f.close()

            cov_kcut = cov[ikmin:ikmax,ikmin:ikmax]
            cov_inv=linalg.inv(cov_kcut)
            #cov_inv= float(nr-nbin-2)*cov_inv/float(nr-1)
            var_inv[isub]=np.diag(cov_inv)
            
        var_inv_mean = np.mean(var_inv,axis=0)
        sigma=np.std(var_inv,axis=0,ddof=1)
        sigma2=sigma*sigma
        fact=np.sum(var_inv_mean*var_inv_mean)
        trace_sigma2[i]=np.sum(sigma2)/fact

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION VARIANCE --------------------------- #
def inv_cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Relative variance of the inverse of the covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and inverse covariance relative variance
    """
    
    simset = DeusPurSet("all_256")
    i=0 #index on list_nr
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)
    
    power_k,dummy = power_spectrum(powertype,mainpath,simset,1,noutput,aexp)
    nbin = power_k.size
    
    sigma2_all = np.zeros((len(list_nr),nbin,nbin))
    
    for nr in list_nr:
        
        nsub = 12288/nr
        inv_cov_sub = np.zeros((nsub,nbin,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%power_pcov[ik,jk])+" ")
                        f.write("\n")
                    f.close()
            inv_cov_sub[isub]=linalg.inv(power_pcov)
    
        inv_cov_mean = np.mean(inv_cov_sub,axis=0)
        sigma = np.std(inv_cov_sub,axis=0,ddof=1)
        sigma2 = sigma*sigma
        fact = (nr-nbin-2)*np.outer(np.diag(inv_cov_mean),np.diag(inv_cov_mean))+(nr-nbin)*inv_cov_mean*inv_cov_mean
        sigma2_all[i]=sigma2/fact
        
        i+=1
    return list_nr, sigma2_all
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX BIAS --------------------------- #
def inv_cov_bias(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [300,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Relative bias of the inverse of the covariance estimator using all_256 Deus Pur set.
        
    Paramters
    ---------
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute inverse covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and inverse covariance relative bias
    """

    simset = DeusPurSet("all_256")
    bias=np.zeros(len(list_nr))
    i=0
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)

    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset,1,simset.nsimmax+1,noutput,aexp)
    cov_inv_all=linalg.inv(power_pcov)
    trace_all=np.trace(cov_inv_all)
    nbin = power_k.size

    for nr in list_nr:
        nsub = 12288/nr
        var_inv = np.zeros((nsub,nbin))
        if (okprint):
            totsim = 12288 - int(math.fmod(12288,nr))
            print nr,totsim

        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                cov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%cov[ik,jk])+" ")
                        f.write("\n")
                    f.close()

            cov_inv=linalg.inv(cov)
            var_inv[isub]=np.diag(cov_inv)

        var_inv_mean = np.mean(var_inv,axis=0)
        trace=np.sum(var_inv_mean)

        bias[i]=trace/trace_all
        i+=1
    return list_nr, bias
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX BIAS WITH k CUT --------------------------- #
def inv_cov_bias_kcut(kmin=0.03,kmax=1.,powertype = "power", mainpath = "", noutput = 1 , aexp = 1., list_nr = [300,500,700,1000,3000,5000,6000], okprint=False, store=True):
    """ Relative bias of the inverse of the covariance estimator using all_256 Deus Pur set cut at given k_min and k_max.
        
    Paramters
    ---------
    kmin: float
        minimum k (default 0.03 h/Mpc)
    kmax: float
        maximum k (default 1 h/Mpc)
    powertype: string
        type of power spectrum: power, nyquist, renormalized, corrected, mcorrected, linear or linear_mock (default power)
    mainpath: string
        path to base folder (default empty)
    noutput: int
        snapshot number (default 1)
    aexp: float
        expansion factor (default 0)
    list_nr: list
        list of realisation numbers at which to compute inverse covariance stats
    okprint: bool
        verbose (default False)
    store: bool
        store intermediate file. If True and files exist they will be overwritten (default True)
        
    Returns
    -------
    1 list and 1 numpy array
        list of realization numbers and inverse covariance relative bias
    """
    
    simset = DeusPurSet("all_256")
    bias=np.zeros(len(list_nr))
    i=0
    
    folder="tmp/"+str("%05d"%noutput)+"/"
    command="mkdir -p tmp; mkdir -p "+folder
    os.system(command)

    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset,1,simset.nsimmax+1,noutput,aexp)
    ikmin = np.searchsorted(power_k,kmin)
    ikmax = np.searchsorted(power_k,kmax)
    power_k_cut = power_k[ikmin:ikmax]
    nbin = power_k_cut.size
    cov_all_kcut = power_pcov[ikmin:ikmax,ikmin:ikmax]
    cov_inv_all=linalg.inv(cov_all_kcut)
    trace_all=np.trace(cov_inv_all)
    
    for nr in list_nr:
        nsub = 12288/nr
        var_inv = np.zeros((nsub,nbin))
        if (okprint):
            totsim = 12288 - int(math.fmod(12288,nr))
            print nr,totsim
    
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename=folder+"cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                if (okprint):
                    print "Reading file: ",filename
                cov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp)
                if (store):
                    if (okprint):
                        print "Storing file: ",filename
                    fltformat="%-.12e"
                    f = open(filename, "w")
                    for ik in xrange(0, nbin):
                        for jk in xrange(0, nbin):
                            f.write(str(fltformat%cov[ik,jk])+" ")
                        f.write("\n")
                    f.close()
        
            cov_kcut=cov[ikmin:ikmax,ikmin:ikmax]
            cov_inv=linalg.inv(cov_kcut)
            var_inv[isub]=np.diag(cov_inv)

        var_inv_mean = np.mean(var_inv,axis=0)
        trace=np.sum(var_inv_mean)
        
        bias[i]=trace/trace_all
        i+=1
    return list_nr, bias
# ---------------------------------------------------------------------------- #



# ------------------ STEIN ESTIMATOR --------------------- #
def stein_estimator(cov, precision, nsim=1, nbin=1, biased_precision=True):
    """ Stein estimator
        
    Parameters
    ----------
    cov: numpy array
        covariance
    precision: numpy array
        inverse covariance
    nsim: int
        number of simulations (default 1)
    nbin: int
        number of bins (default 1)
    biased_precision: bool
        use Hartlap correction for inverse covariance (default True)
        
    Returns
    ------
    numpy array
        Stein estimator
    """
    
    if (biased_precision):
        stein = (nsim-nbin-2.)/(nsim-1.)*precision + (nbin*(nbin+1)-2.)/((nsim-1.)*np.trace(cov))*np.eye(nbin)
    else:
        stein = precision + (nbin*(nbin+1)-2.)/((nsim-1.)*np.trace(cov))*np.eye(nbin)
    return stein
