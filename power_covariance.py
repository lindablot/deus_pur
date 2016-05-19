#!/usr/bin/env python
# power_covariance.py - Linda Blot (linda.blot@obspm.fr) - 2013
# ---------------------------------- IMPORT ---------------------------------- #
import math
import glob
import os
import sys
import numpy as np
import scipy.interpolate as itl
from scipy import stats
from numpy import linalg
from utilities import *
from read_files import *
from power_types import *
from power_stats import *
#import os.path
# ---------------------------------------------------------------------------- #



# -------------------------------- COVARIANCE POWER -------------------------------- #
def cov_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nsim = isimmax-isimmin

    fname = file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
    fltformat="%-.12e"

    if (os.path.isfile(fname)):
        if (okprint):
            print "Reading power spectrum covariance from file: ",fname
        power_k, power_pmean, power_psigma = mean_power(powertype,mainpath,simset.name,isimmin,isimmax,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
        power_pcov = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing power spectrum covariance"
        if (powertype=="linear"):
            power_k, power_pmean = power_spectrum("linear",mainpath,simset.name,1,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
            power_psigma = np.sqrt(2./simset.N_k(power_k))*power_pmean
            power_pcov = np.diag(power_psigma*power_psigma)
        else:
            power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset.name, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, nmodel, okprint)
            diff_power_p = np.zeros(power_k.size)
            power_pcov = np.zeros((power_k.size,power_k.size))
            for isim in xrange(isimmin, isimmax):
                isim0 = isim - isimmin
                true_simset,true_isim = sim_iterator(simset.name, isim)
                if (okprint) :
                    print true_simset,true_isim
                dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
                diff_power_p = power_p - power_pmean
                power_pcov += np.outer(diff_power_p,diff_power_p)
            power_pcov/=float(nsim-1)

        if (store):
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat%power_pcov[i,j])+" ")
                f.write("\n")
            f.close()

    return power_k, power_pmean, power_psigma, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- POWER SPECTRUM COVARIANCE WITH k CUT ------------------------- #
def cov_power_kcut(kmin, kmax, powertype, mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nsim = isimmax-isimmin
    
    fname = file_name("cov",powertype,simset,isimmin,isimmax,noutput,nmodel)
    fltformat="%-.12e"

    if (os.path.isfile(fname)):
        if (okprint):
            print "Reading power spectrum covariance from file: ",fname
        power_k, power_pmean, power_psigma = mean_power(powertype,mainpath,simset.name,isimmin,isimmax,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
        idx=[(power_k > kmin) & (power_k < kmax)]
        power_k_kcut=power_k[idx]
        power_pmean_kcut = power_pmean[idx]
        power_psigma_kcut = power_psigma[idx]
        power_pcov = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing power spectrum covariance"
        nsim = isimmax - isimmin
        power_k, power_pmean, power_psigma, power_pcov_nocut = cov_power(powertype,mainpath,simset.name,isimmin,isimmax,noutput,aexp,growth_a,growth_dplus,nmodel,okprint)
        idx=[(power_k > kmin) & (power_k < kmax)]
        power_k_kcut=power_k[idx]
        power_pmean_kcut = power_pmean[idx]
        power_psigma_kcut = power_psigma[idx]
        imin=np.where(power_k==power_k_kcut[0])[0]
        imax=np.where(power_k==power_k_kcut[power_k_kcut.size-1])[0]+1
        power_pcov = power_pcov_nocut[imin:imax,imin:imax]

        if (store):
            f = open(fname, "w")
            for i in xrange(0, power_k_cut.size):
                for j in xrange(0, power_k_cut.size):
                    f.write(str(fltformat%power_pcov[i,j])+" ")
                f.write("\n")
            f.close()

    return power_k_kcut, power_pmean_kcut, power_psigma_kcut, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- CORRELATION COEFFICIENT POWER ------------------------- #
def corr_coeff(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nsim = isimmax-isimmin

    fname = file_name("corr_coeff",powertype,simset,isimmin,isimmax,noutput,nmodel)
    fltformat="%-.12e"
    if (os.path.isfile(fname)):
        if (okprint):
            print "Reading power spectrum correlation coefficient from file: ",fname
        corr_coeff = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing power spectrum covariance"
        power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset.name, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, nmodel, okprint)
        norm = np.outer(power_psigma,power_psigma)
        corr_coeff = power_pcov/norm

        if (store):
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat%corr_coeff[i,j])+" ")
                f.write("\n")
            f.close()

    return corr_coeff
# ---------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE ------------------------------ #
def signoise(powertype = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), nmodel = 0, unbiased = False, okprint = False, store = False):
    
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    
    fname = file_name("sig_noise",powertype,simset,isimmin,isimmax,noutput,nmodel)
    fltformat="%-.12e"

    if(os.path.isfile(fname)):
        if (okprint):
            print "Reading signal-to-noise from file: ",fname
        kmax, sig_noise = np.loadtxt(fname,unpack=True)
    else:
        if (okprint):
            print "Computing signal-to-noise"
        nmax = nsimmax-1
        step = 1
        
        power_k, power_pmean, dummy, power_pcov = cov_power(powertype, mainpath, simset.name, 1, nsimmax, noutput, aexp, growth_a, growth_dplus, nmodel, okprint)
        
        kmin=power_k[0]
        num_iter=min(power_k.size,nmax-2)
        sig_noise = np.zeros(num_iter/step)
        kmax = np.zeros(num_iter/step)
        for ikmax in xrange(0,num_iter/step):
            ikk=ikmax*step+1
            kmax[ikmax]=power_k[ikk]
            idx=(power_k < kmax[ikmax])
            power_p_new=power_pmean[idx]
            power_kk = power_k[idx]
            power_pcov_kmax=power_pcov[0:ikk,0:ikk]
            cov_inv=linalg.inv(power_pcov_kmax)
            if (unbiased):
                cov_inv= float(nmax-num_iter-2)*cov_inv/float(nmax-1)
            sig_noise[ikmax]=math.sqrt(np.dot(power_p_new.T,np.dot(cov_inv,power_p_new)))
        

        if (store):
            f = open(fname, "w")
            for i in xrange(0, kmax.size):
                f.write(str(fltformat%kmax[i])+" "+str(fltformat%sig_noise[i])+"\n")
            f.close()

    return kmax, sig_noise
# --------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE AT A GIVEN K ------------------------------ #
def signoise_k(powertype = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), kmax=0.4, unbiased = False, okprint = False):
    if (type(simset) is str):
        simset=DeusPurSet(simset)
    nmax = nsimmax-1
    
    power_k, power_pmean, power_psigma, power_pcov = cov_power_kcut(0., kmax, powertype, mainpath, simset.name, 1, nsimmax, noutput, aexp, growth_a, growth_dplus,okprint)
    cov_inv=linalg.inv(power_pcov)
    num_iter=min(power_k.size,nmax-2)
    if (unbiased):
        cov_inv= float(nmax-num_iter-2)*cov_inv/float(nmax-1)
    sig_noise=math.sqrt(np.dot(power_pmean.T,np.dot(cov_inv,power_pmean)))
    
    return sig_noise
# --------------------------------------------------------------------------- #