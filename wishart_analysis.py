#!/usr/bin/env python
# wishart_analysis.py - Linda Blot (linda.blot@obspm.fr) - 2014
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
from power_covariance import *
from math import sqrt as sqrt
#import os.path
# ---------------------------------------------------------------------------- #



# --------------------------- TRACE OF SAMPLE COVARIANCE VARIANCE --------------------------- #
def cov_variance_trace(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    nbin = power_k.size

    for nr in list_nr:

        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        nsim = 0
        
        print nr

        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/sigma_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            filename_cov="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                power_psigma=np.loadtxt(filename, unpack=True)
            elif(os.path.isfile(filename_cov)):
                power_pcov=np.loadtxt(filename_cov, unpack=True)
                power_psigma=np.sqrt(np.diag(power_pcov))
            else:
                dummy,dummy,power_psigma=mean_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, nbin):
                    f.write(str(fltformat%power_psigma[ik])+"\n")
                f.close()

            power_pvar[isub]=power_psigma**2

        var_mean = np.zeros(nbin)
        for ik in xrange(0,nbin):
            var_mean[ik]=np.mean(power_pvar[:,ik])

        sigma2 = np.zeros(nbin)
        fact = 0.
        for ik in xrange(0,nbin):
            for isubset in xrange(0,nsub):
                sigma2[ik]+=(power_pvar[isubset,ik] - var_mean[ik]) * (power_pvar[isubset,ik] - var_mean[ik])
            fact+=var_mean[ik]*var_mean[ik]
            trace_sigma2[i]+=sigma2[ik]
        trace_sigma2[i]/=(fact*float(nsub-1))

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- DET OF SAMPLE COVARIANCE VARIANCE --------------------------- #
def cov_variance_det(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):
    
    simset = "all_256"
    det_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    nbin = power_k.size
    
    for nr in list_nr:
        
        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        cov_sub = np.zeros((nsub,nbin,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, nbin):
                    for jk in xrange(0, nbin):
                        f.write(str(fltformat%power_pcov[ik,jk])+" ")
                    f.write("\n")
                f.close()
                    
            cov_sub[isub]=power_pcov
    
        cov_mean = np.zeros((nbin,nbin))
        for ik in xrange(0,nbin):
            for jk in xrange(0,nbin):
                cov_mean[ik,jk]=np.mean(cov_sub[:,ik,jk])
                    
        sigma2 = np.zeros((nbin,nbin))
        rhs = np.zeros((nbin,nbin))
        for ik in xrange(0,nbin):
            for jk in xrange(0,nbin):
                for isubset in xrange(0,nsub):
                    sigma2[ik,jk]+=(cov_sub[isubset,ik,jk] - cov_mean[ik,jk]) * (cov_sub[isubset,ik,jk] - cov_mean[ik,jk])
                rhs[ik,jk]=(cov_mean[ik,ik]*cov_mean[jk,jk]+cov_mean[ik,jk]**2)
        sigma2/=float(nsub-1)
        
        det_sigma2[i]=np.linalg.det(sigma2)/np.linalg.det(rhs)
        
        i+=1
    return list_nr, det_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- COVARIANCE VARIANCE --------------------------- #
def cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):
    
    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    nbin = power_k.size
    
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
            filename="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                power_pcov=np.loadtxt(filename, unpack=True)
            else:
                dummy,dummy,dummy,power_pcov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, nbin):
                    for jk in xrange(0, nbin):
                        f.write(str(fltformat%power_pcov[ik,jk])+" ")
                    f.write("\n")
                f.close()
            
            cov_sub[isub]=power_pcov
        
        cov_mean = np.zeros((nbin,nbin))
        for ik in xrange(0,nbin):
            for jk in xrange(0,nbin):
                cov_mean[ik,jk]=np.mean(cov_sub[:,ik,jk])
    
        sigma2 = np.zeros((nbin,nbin))
        fact = 0.
        for ik in xrange(0,nbin):
            for jk in xrange(0,nbin):
                for isubset in xrange(0,nsub):
                    sigma2[ik,jk]+=(cov_sub[isubset,ik,jk] - cov_mean[ik,jk]) * (cov_sub[isubset,ik,jk] - cov_mean[ik,jk])
                sigma2[ik,jk]/=(cov_mean[ik,ik]*cov_mean[jk,jk]+cov_mean[ik,jk]**2)
        sigma2/=float(nsub-1)

        sigma2_all[i]=sigma2
        
        i+=1
    return list_nr, sigma2_all
# ---------------------------------------------------------------------------- #



# --------------------------- SAMPLE COVARIANCE VARIANCE WITH k CUT --------------------------- #
def cov_variance_kcut(kmin=0.03, kmax = 1., powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [10,50,100,500,700,1000,3000,5000,6000], okprint=False):
    
    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0 #index on list_nr
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    index = [(power_k > kmin) & (power_k < kmax)]
    power_k_cut = power_k[index]
    nbin = power_k_cut.size
    
    for nr in list_nr:
        
        nsub = 12288/nr
        power_pmean = np.zeros((nsub,nbin))
        power_pvar = np.zeros((nsub,nbin))
        
        print nr
        
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/sigma_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            filename_cov="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                power_psigma=np.loadtxt(filename, unpack=True)
            elif(os.path.isfile(filename_cov)):
                power_pcov=np.loadtxt(filename_cov, unpack=True)
                power_psigma=np.sqrt(np.diag(power_pcov))
            else:
                dummy,dummy,power_psigma=mean_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, power_k.size):
                    f.write(str(fltformat%power_psigma[ik])+"\n")
                f.close()
            
            power_psigma_kcut = power_psigma[index]
            power_pvar[isub]=power_psigma_kcut**2
                        
        var_mean = np.zeros(nbin)
        for ik in xrange(0,nbin):
            var_mean[ik]=np.mean(power_pvar[:,ik])

    
        sigma2 = np.zeros(nbin)
        fact = 0.
        for ik in xrange(0,nbin):
            for isubset in xrange(0,nsub):
                sigma2[ik]+=(power_pvar[isubset,ik] - var_mean[ik]) * (power_pvar[isubset,ik] - var_mean[ik])
            fact+=var_mean[ik]*var_mean[ik]
            trace_sigma2[i]+=sigma2[ik]
        trace_sigma2[i]/=(fact*float(nsub-1))
        
        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX VARIANCE --------------------------- #
def inv_cov_variance(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [300,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0

    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)

    for nr in list_nr:
        nsub = 12288/nr
        nbin = power_k.size
        var_inv = np.zeros((nsub,nbin))
        totsim = 12288 - int(math.fmod(12288,nr))
        
        print nr,totsim

        var_all = np.zeros(nbin)
        var_inv_mean = np.zeros(nbin)
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                cov=np.loadtxt(filename, unpack=True)
            else:
                kcov,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, nbin):
                    for jk in xrange(0, nbin):
                        f.write(str(fltformat%cov[ik,jk])+" ")
                    f.write("\n")
                f.close()

            cov_inv=linalg.inv(cov)
            #cov_inv= float(nr-nbin-2)*cov_inv/float(nr-1)
            
            for ik in xrange(0,nbin):
                var_inv[isub,ik] = cov_inv[ik,ik]
                var_inv_mean[ik] += var_inv[isub,ik]
        var_inv_mean /= float(nsub)
        
        fact=0.
        sigma2=np.zeros(nbin)
        for ik in xrange(0,nbin):
            for isub in xrange(0,nsub):
                sigma2[ik]+=(var_inv[isub,ik] - var_inv_mean[ik]) * (var_inv[isub,ik] - var_inv_mean[ik])
            fact += var_inv_mean[ik] * var_inv_mean[ik]
            trace_sigma2[i]+=sigma2[ik]

        trace_sigma2[i]/=(fact*float(nsub-1))

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX VARIANCE --------------------------- #
def inv_cov_variance_kcut(kmin = 0.03, kmax = 1., powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [300,500,700,1000,3000,5000,6000], okprint=False):
    
    simset = "all_256"
    trace_sigma2=np.zeros(len(list_nr))
    i=0
    
    power_k,dummy = power_spectrum(powertype,mainpath,"4096_adaphase_256",1,noutput,aexp,growth_a,growth_dplus)
    index = [(power_k < kmax) & (power_k > kmin)]
    power_k_kcut = power_k[index]
    nbin = power_k_kcut.size
    
    aa = np.arange(0,power_k.size-1)
    iks = aa[index]
    ikmin = iks[0]
    ikmax = iks[iks.size-1]
    
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
            filename="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                cov=np.loadtxt(filename, unpack=True)
            
            else:
                dummy,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
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
            
            for ik in xrange(0,nbin):
                var_inv[isub,ik] = cov_inv[ik,ik]
                var_inv_mean[ik] += var_inv[isub,ik]
        var_inv_mean /= float(nsub)
        
        fact=0.
        sigma2=np.zeros(nbin)
        for ik in xrange(0,nbin):
            for isub in xrange(0,nsub):
                sigma2[ik]+=(var_inv[isub,ik] - var_inv_mean[ik]) * (var_inv[isub,ik] - var_inv_mean[ik])
            fact += var_inv_mean[ik] * var_inv_mean[ik]
            trace_sigma2[i]+=sigma2[ik]
    
        trace_sigma2[i]/=(fact*float(nsub-1))

        i+=1
    return list_nr, trace_sigma2
# ---------------------------------------------------------------------------- #



# --------------------------- PRECISION MATRIX BIAS --------------------------- #
def inv_cov_bias(powertype = "power", mainpath = "", noutput = 1 , aexp = 1., growth_a = np.zeros(0), growth_dplus = np.zeros(0), list_nr = [300,500,700,1000,3000,5000,6000], okprint=False):

    simset = "all_256"
    bias=np.zeros(len(list_nr))
    i=0

    power_k, power_pmean, power_psigma, power_pcov = cov_power(powertype, mainpath, simset,1,12289,noutput,aexp,growth_a,growth_dplus)
    cov_inv_all=linalg.inv(power_pcov)
    trace_all=np.trace(cov_inv_all)

    for nr in list_nr:
        nsub = 12288/nr
        nbin = power_k.size
        if (okprint):
            totsim = 12288 - int(math.fmod(12288,nr))
            print nr,totsim

        var_inv_mean = np.zeros(nbin)
        for isub in xrange(0, nsub):
            isimmin = isub * nr + 1
            isimmax = (isub+1)*nr
            filename="tmp/"+str("%05d"%noutput)+"/cov_"+powertype+"_"+str("%05d"%nr)+"_"+str("%05d"%isub)+".txt"
            if(os.path.isfile(filename)):
                cov=np.loadtxt(filename, unpack=True)
            else:
                kcov,dummy,dummy,cov=cov_power(powertype, mainpath, simset, isimmin, isimmax+1, noutput, aexp, growth_a, growth_dplus)
                fltformat="%-.12e"
                f = open(filename, "w")
                for ik in xrange(0, nbin):
                    for jk in xrange(0, nbin):
                        f.write(str(fltformat%cov[ik,jk])+" ")
                    f.write("\n")
                f.close()

            cov_inv=linalg.inv(cov)

            for ik in xrange(0,nbin):
                var_inv_mean[ik]+=cov_inv[ik,ik]
        var_inv_mean /= float(nsub)

        trace=np.sum(var_inv_mean)

        bias[i]=trace/trace_all
        i+=1
    return list_nr, bias
# ---------------------------------------------------------------------------- #



# ------------------ STEIN ESTIMATOR --------------------- #
def stein_estimator(cov=np.zeros((0,0)),precision=np.zeros((0,0)),nsim=1,nbin=1,biased_precision=True):
    if (biased_precision):
        stein = (nsim-nbin-2.)/(nsim-1.)*precision + (nbin*(nbin+1)-2.)/((nsim-1.)*np.trace(cov))*np.eye(nbin)
    else:
        stein = precision + (nbin*(nbin+1)-2.)/((nsim-1.)*np.trace(cov))*np.eye(nbin)
    return stein
