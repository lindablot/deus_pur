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
from read_files import *
from power_types import *
from power_stats import *
from math import sqrt as sqrt
#import os.path
# ---------------------------------------------------------------------------- #



# -------------------------------- COVARIANCE POWER -------------------------------- #
def cov_power(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):
    
    nsim = isimmax-isimmin
    
    if ((simset=="all_256" and nsim==12288) or (simset=="all_1024" and nsim==96)):
        fname = "cov_"+simset+"_"+powertype+"_"+str("%05d"%noutput)+".txt"
        if(os.path.isfile(fname)):
            power_pcov = np.loadtxt(fname,unpack=True)
            if (simset=="all_256"):
                power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
            elif (simset=="all_1024"):
                power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
        
        else:
            if (simset=="all_256"):
                power_k, power_pmean, power_psigma = mean_power_all(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
            else:
                power_k, power_pmean, power_psigma = mean_power_all_1024(powertype, mainpath, noutput, aexp, growth_a, growth_dplus, okprint)
                
            power_pcov = np.zeros((power_k.size,power_k.size))
                
            for isim in xrange(1, nsim+1):
                true_simset,true_isim = sim_iterator(simset, isim)
                if (okprint) :
                    current_file = file_path("power", mainpath, iset, isim, noutput)
                    print current_file
                dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
                for ik in xrange(0, power_k.size):
                    for jk in xrange(0, power_k.size):
                        power_pcov[ik,jk] += (power_p[ik]-power_pmean[ik])*(power_p[jk]-power_pmean[jk])

            power_pcov /= float(nsim-1)

            fltformat="%-.12e"
            f = open(fname, "w")
            for i in xrange(0, power_k.size):
                for j in xrange(0, power_k.size):
                    f.write(str(fltformat%power_pcov[i,j])+" ")
                f.write("\n")
            f.close()
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
        power_pcov = np.zeros((power_k.size,power_k.size))
        
        for isim in xrange(isimmin, isimmax):
            true_simset,true_isim = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, true_simset, true_isim, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,true_simset,true_isim,noutput,aexp,growth_a,growth_dplus,okprint)
            for ik in xrange(0, power_k.size):
                for jk in xrange(0, power_k.size):
                    power_pcov[ik,jk] += (power_p[ik]-power_pmean[ik])*(power_p[jk]-power_pmean[jk])

        power_pcov /= float(nsim-1)

    return power_k, power_pmean, power_psigma, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- POWER SPECTRUM COVARIANCE WITH k CUT ------------------------- #
def cov_power_kcut(kmin, kmax, powertype, mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), sampling=0, okprint = True):
    
    nsim = isimmax - isimmin
    if (simset=="all_256" and nsim==12288):
        power_k, power_pmean, power_psigma, power_pcov_nocut = cov_power(powertype,mainpath,simset,noutput,aexp,growth_a,growth_dplus)
        idx=[(power_k > kmin) & (power_k < kmax)]
        power_k_kcut=power_k[idx]
        power_pmean_kcut = power_pmean[idx]
        power_psigma_kcut = power_psigma[idx]
        imin=np.where(power_k==power_k_kcut[0])[0]
        imax=np.where(power_k==power_k_kcut[power_k_kcut.size-1])[0]+1
        power_pcov = power_pcov_nocut[imin:imax,imin:imax]
    
    else:
        power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
        idx=[(power_k > kmin) & (power_k < kmax)]
        power_k_kcut = power_k[idx]
        power_pmean_kcut = power_pmean[idx]
        power_psigma_kcut = power_psigma[idx]

        power_pcov = np.zeros((power_k_kcut.size,power_k_kcut.size))
        for isim in xrange(isimmin, isimmax):
            iset, isimu = sim_iterator(simset, isim)
            if (okprint) :
                current_file = file_path("power", mainpath, iset, isimu, noutput)
                print current_file
            dummy, power_p = power_spectrum(powertype,mainpath,iset,isimu,noutput,aexp,growth_a,growth_dplus,okprint)
            power_p_kcut = power_p[idx]
            for ik in xrange(0, power_k_kcut.size):
                for jk in xrange(0, power_k_kcut.size):
                    power_pcov[ik,jk] += (power_p_kcut[ik]-power_pmean_kcut[ik])*(power_p_kcut[jk]-power_pmean_kcut[jk])
        power_pcov /= float(nsim-1)

    return power_k_kcut, power_pmean_kcut, power_psigma_kcut, power_pcov
# ---------------------------------------------------------------------------- #



# -------------------- CORRELATION COEFFICIENT POWER ------------------------- #
def corr_coeff(powertype = "power", mainpath = "", simset = "", isimmin = 1, isimmax = 2, noutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), okprint = False):

    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset, isimmin, isimmax, noutput, aexp, growth_a, growth_dplus, okprint)
    
    corr_coeff = np.zeros((power_k.size,power_k.size))
    for ik in xrange(0, power_k.size):
        for jk in xrange(0, power_k.size):
            corr_coeff[ik,jk] = power_pcov[ik,jk]/(sqrt(power_pcov[ik,ik])*sqrt(power_pcov[jk,jk]))#/(power_psigma[ik]*power_psigma[jk])
            
    return corr_coeff
# ---------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE ------------------------------ #
def signoise(powertype = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), unbiased = False, okprint = False):
    
    fname="sig_noise_"+powertype+"_"+simset+"_"+str("%05d"%noutput)+".txt"
    if(os.path.isfile(fname) and simset=="all_256" and nsimmax==12289):
        kmax, sig_noise = np.loadtxt(fname,unpack=True)
    else:
        nmax = nsimmax-1
        step = 1
        
        power_k, dummy, dummy, power_pcov = cov_power(powertype, mainpath,simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus)
        dummy, power_pmean, dummy = mean_power_all("mcorrected",mainpath,noutput,aexp,growth_a,growth_dplus)
        
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
            signoise2=0.
            for ik in xrange(0,power_kk.size):
                for jk in xrange(0,power_kk.size):
                    signoise2+=power_p_new[ik]*cov_inv[ik,jk]*power_p_new[jk]
            sig_noise[ikmax]=sqrt(signoise2)
        
        fltformat = "%-.12e"
        f = open(fname, "w")
        for i in xrange(0, kmax.size):
            f.write(str(fltformat%kmax[i])+" "+str(fltformat%sig_noise[i])+"\n")
        f.close()

    return kmax, sig_noise
# --------------------------------------------------------------------------- #



# ----------------------------- SIGNAL TO NOISE AT A GIVEN K ------------------------------ #
def signoise_k(powertype = "nyquist", mainpath = "", simset = "", noutput = 1, nsimmax = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), kmax=0.4, unbiased = False, okprint = False):

    nmax = nsimmax-1
    
    power_k, power_p = power_spectrum(mainpath, simset, 1, noutput, aexp, growth_a, growth_dplus)
    power_k, power_pmean, power_psigma = mean_power(powertype, mainpath, simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus)
    
    num_iter=min(power_k.size,nmax-2)
    idx=(power_k < kmax)
    power_p_new=power_p[idx]
    power_kk,power_pcov=cov_power_kcut(0., kmax, powertype, mainpath, simset, 1, nsimmax, noutput, aexp, growth_a, growth_dplus, 0, okprint)
    cov_inv=linalg.inv(power_pcov)
    if (unbiased):
        cov_inv= float(nmax-num_iter-2)*cov_inv/float(nmax-1)
    signoise2=0.
    for ik in xrange(0,power_kk.size):
        for jk in xrange(0,power_kk.size):
            signoise2+=power_p_new[ik]*cov_inv[ik,jk]*power_p_new[jk]
    sig_noise=sqrt(signoise2)
    
    return sig_noise
# --------------------------------------------------------------------------- #



# -------------------------------- BACKUP COVARIANCE ------------------------------- #
def backup_cov(powertype = "nyquist", mainpath = "", simset = "", ioutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), fltformat = "%-.12e", okprint = False):
    if (simset == "4096_furphase" or simset == "4096_otherphase"):
        isimmax = 1
        noutputmax = 31
    elif (simset == "4096_furphase_512"):
        isimmax = 512
        noutputmax = 9
    elif (simset == "64_adaphase_1024"):
        isimmax = 64
        noutputmax = 9
    elif (simset == "64_curiephase_1024"):
        isimmax = 32
        noutputmax = 9
    elif (simset == "all_256"):
        isimmax = 12288
        noutputmax = 9
    elif (simset == "all_1024"):
        isimmax = 96
        noutputmax = 9
    else:
        isimmax = 4096
        noutputmax = 9
    #for ioutput in xrange(noutputmax, noutputmax+1):
    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus, okprint)

    fname = "cov_"+simset+"_"+powertype+"_"+str("%05d"%ioutput)+".txt"
    f = open(fname, "w")
    for i in xrange(0, power_k.size):
        for j in xrange(0, power_k.size):
            f.write(str(fltformat%power_pcov[i][j])+" ")
        f.write("\n")
    f.close()
    return
# ---------------------------------------------------------------------------- #



# ------------- BACKUP CORRELATION COEFFICIENT ------------------------------- #
def backup_corr(powertype="power", mainpath = "", simset = "", ioutput = 1, aexp = 0., growth_a = np.zeros(0), growth_dplus = np.zeros(0), fltformat = "%-.12e"):
    if (simset == "4096_furphase" or simset == "4096_otherphase"):
        isimmax = 1
        noutputmax = 31
        sim_set = simset
    elif (simset == "4096_furphase_512"):
        isimmax = 512
        noutputmax = 9
        sim_set = simset
    elif (simset == "64_adaphase_1024"):
        isimmax = 64
        noutputmax = 9
        sim_set = simset
    elif (simset == "64_curiephase_1024"):
        isimmax = 32
        noutputmax = 9
    elif (simset == "all_256"):
        isimmax = 12288
        noutputmax = 9
    elif (simset == "all_1024"):
        isimmax = 96
        noutputmax = 9
    else:
        isimmax = 4096
        noutputmax = 9

    #for ioutput in range(noutputmax-1,noutputmax):
    fname_cov="cov_"+simset+"_"+powertype+"_"+str("%05d"%ioutput)+".txt"
    fname_corr="corr_coeff_"+powertype+"_"+simset+"_"+str("%05d"%ioutput)+".txt"
    if (os.path.isfile(fname_cov)):
        corr_coeffi = corr_coeff_b(powertype, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus, fname_cov)
    else:
        corr_coeffi = corr_coeff(powertype, mainpath, simset, 1, isimmax+1, ioutput, aexp, growth_a, growth_dplus)

    f = open(fname_corr, "w")
    for i in xrange(0, corr_coeffi.shape[0]):
        for j in xrange(0, corr_coeffi.shape[0]):
            #print i, j, corr_coeffi[i][j]
            f.write(str(fltformat%corr_coeffi[i][j])+" ")
        f.write("\n")
    f.close()
    return
# ---------------------------------------------------------------------------- #
